// Block-causal attention over one packed sequence: prefix tokens (seq 0) then
// branch tokens (seq 1..). A query sees a key iff the key is in the prefix or
// in the query's own branch, is at or before the query's position, and (on
// sliding-window layers) within the window. That is the isolation rule the
// llama.cpp backend gets from its per-sequence KV cache. When several
// requests share a pass, each has its own prefix and branches under a group
// number in the high bits of the sequence id (engine.rs SEQ_GROUP_SHIFT) and
// sees nothing of the others.
//
// `kv_heads` KV heads, each shared by `heads / kv_heads` consecutive query
// heads (one for E2B, two for E4B). Q is read from the layer's fused
// projection buffer (row stride q_stride), K and V from a K/V buffer
// [t][kv_heads x HD | kv_heads x HD] (all K heads, then all V heads) that
// may belong to an earlier layer (Gemma 4 shares K/V). The keys are cache
// tokens 0..t; the queries are cache tokens base..t, held at workspace rows
// 0.. (base > 0 when the prefix is resident from an earlier pass and only the
// branches run). A workgroup of ROWS x KB invocations handles ROWS query rows
// of one KV head, wg.y: (ROWS / hpg) tokens x the hpg query heads of that
// group, with their Q held as f16 in workgroup memory, and streams that
// head's keys KB at a time. A tile no row can see is skipped (keys after the block,
// other branches). Otherwise the K tile is staged as f16, each invocation
// computes one full (row, key) score, then the same tile buffer is refilled
// with V and each invocation owns HD/KB output dims of one row for the
// online softmax and P.V. Staging K and V as f16 (one union buffer) and
// reading them as vec4 keeps the kernel off scalar global loads, which is
// what bounded the previous version. HD, ROWS and KB are substituted by the
// engine (16 x 8: 128 invocations, 13 KB at HD 256 / 25 KB at HD 512; see
// attn_tile for the shapes that measured worse).

struct Params { t: u32, heads: u32, window: u32, q_stride: u32, kv_heads: u32, base: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> kv: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tok_meta: array<i32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

const HD: u32 = 256u;
const ROWS: u32 = 16u;     // query rows per workgroup = tokens * heads
const KB: u32 = 16u;       // keys per tile; also invocations per row
const N: u32 = ROWS * KB;  // invocations
const HP: u32 = HD / 2u;   // f16 pairs per head row
const HQ: u32 = HD / 4u;   // vec4 per head row
const CH: u32 = HD / KB;   // output dims per invocation
const NV: u32 = CH / 4u;   // vec4 accumulators per invocation
const NEG: f32 = -1.0e30;

var<workgroup> qs: array<u32, ROWS * HP>;  // [ROWS][HP] f16 pairs
var<workgroup> kvs: array<u32, KB * HP>;   // [KB][HP] f16 pairs: K, then V
var<workgroup> s: array<f32, ROWS * KB>;   // [ROWS][KB] masked scores
var<workgroup> ps: array<f32, ROWS * KB>;  // [ROWS][KB] exp(score - m), 0 if masked
var<workgroup> kpos: array<i32, KB>;
var<workgroup> kseq: array<i32, KB>;
var<workgroup> qpos: array<i32, ROWS>;
var<workgroup> qseq: array<i32, ROWS>;
var<workgroup> qlo: array<i32, ROWS>;
var<workgroup> qhi: array<i32, ROWS>;
var<workgroup> krange: vec2<u32>;
var<workgroup> tile_any: bool;

// Sequence ids: bits 0..12 number the branch within a request (0 = its
// prefix), the bits above number the request within the pass. Same request,
// then prefix or own branch.
fn visible(qp: i32, qs_: i32, kp: i32, ks_: i32) -> bool {
    let same_group = ((ks_ ^ qs_) >> 12u) == 0;
    return same_group && ((ks_ & 0xfff) == 0 || ks_ == qs_) && kp <= qp && (p.window == 0u || u32(qp - kp) < p.window);
}

// Stage KB rows of K (half == 0) or V (half == 1) of KV head g, tile j0, as
// f16 pairs. A kv row is 2 x kv_heads head rows: the K heads, then the V heads.
fn stage_kv(li: u32, j0: u32, half: u32, g: u32) {
    let stride4 = 2u * p.kv_heads * HQ;
    let off4 = (half * p.kv_heads + g) * HQ;
    for (var e = li; e < KB * HQ; e += N) {
        let kk = e / HQ;
        let d4 = e % HQ;
        let j = j0 + kk;
        var v = vec4<f32>(0.0);
        if (j < p.t) { v = kv[j * stride4 + off4 + d4]; }
        kvs[kk * HP + 2u * d4] = pack2x16float(v.xy);
        kvs[kk * HP + 2u * d4 + 1u] = pack2x16float(v.zw);
    }
}

@compute @workgroup_size(128) // ROWS * KB, substituted by the engine
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let heads = p.heads;
    let hpg = heads / p.kv_heads; // query heads per KV head
    let g = wg.y;                 // this workgroup's KV head
    let tb = ROWS / hpg;
    let tok0 = wg.x * tb;

    let row = li / KB;          // 0..ROWS, both roles
    let key = li % KB;          // score role
    let chunk = key * CH;       // output role: CH dims of `row`
    let tok = tok0 + row / hpg;   // workspace row; cache token base + tok
    let head = g * hpg + row % hpg;
    let ok = p.base + tok < p.t;

    // Q tile as f16 pairs: this invocation's CH dims of its row.
    {
        let qb = (tok * p.q_stride + head * HD + chunk) / 4u;
        for (var i = 0u; i < NV; i++) {
            var v = vec4<f32>(0.0);
            if (ok) { v = q[qb + i]; }
            qs[row * HP + chunk / 2u + 2u * i] = pack2x16float(v.xy);
            qs[row * HP + chunk / 2u + 2u * i + 1u] = pack2x16float(v.zw);
        }
    }
    if (li < tb) {
        let tk = p.base + tok0 + li;
        if (tk < p.t) {
            qpos[li] = tok_meta[4u * tk];
            qseq[li] = tok_meta[4u * tk + 1u];
            qlo[li] = tok_meta[4u * tk + 2u];
            qhi[li] = tok_meta[4u * tk + 3u];
        } else {
            qpos[li] = -1;
            qseq[li] = -2;
            qlo[li] = 0x7fffffff;
            qhi[li] = 0;
        }
    }
    workgroupBarrier();
    // Keys this workgroup's tokens can see lie in cache rows [lo, hi): from
    // the first row of their request to the last of these tokens. Tiles
    // outside are never visited, so a pass holding several requests costs
    // each of them its own length, not the whole cache.
    if (li == 0u) {
        var lo = 0x7fffffff;
        var hi = 0;
        for (var qi = 0u; qi < tb; qi++) {
            lo = min(lo, qlo[qi]);
            hi = max(hi, qhi[qi]);
        }
        krange = vec2<u32>(u32(max(lo, 0)), u32(max(hi, 0)));
    }
    let kr = workgroupUniformLoad(&krange);

    var m = NEG;
    var l = 0.0;
    var o: array<vec4<f32>, NV>;
    for (var i = 0u; i < NV; i++) { o[i] = vec4<f32>(0.0); }

    let ntiles = (min(kr.y, p.t) + KB - 1u) / KB;
    for (var tile = kr.x / KB; tile < ntiles; tile++) {
        let j0 = tile * KB;
        if (li < KB) {
            let j = j0 + li;
            if (j < p.t) {
                kpos[li] = tok_meta[4u * j];
                kseq[li] = tok_meta[4u * j + 1u];
            } else {
                kpos[li] = 0x7fffffff;
                kseq[li] = -3;
            }
        }
        workgroupBarrier();
        // Skip the tile when no row can see any of its keys. One invocation
        // decides and the value is broadcast with workgroupUniformLoad, so
        // the branch is provably uniform (Tint insists).
        if (li == 0u) {
            var any = false;
            for (var k = 0u; k < KB; k++) {
                for (var qi = 0u; qi < tb; qi++) {
                    any = any || visible(qpos[qi], qseq[qi], kpos[k], kseq[k]);
                }
            }
            tile_any = any;
        }
        if (!workgroupUniformLoad(&tile_any)) {
            continue;
        }
        stage_kv(li, j0, 0u, g);
        workgroupBarrier();

        // One full score per invocation.
        {
            // Four independent partial sums: one chain of HP dependent adds
            // is latency-bound.
            var a0 = 0.0;
            var a1 = 0.0;
            var a2 = 0.0;
            var a3 = 0.0;
            let qb = row * HP;
            let kb = key * HP;
            for (var i = 0u; i < HP; i += 4u) {
                a0 += dot(unpack2x16float(qs[qb + i]), unpack2x16float(kvs[kb + i]));
                a1 += dot(unpack2x16float(qs[qb + i + 1u]), unpack2x16float(kvs[kb + i + 1u]));
                a2 += dot(unpack2x16float(qs[qb + i + 2u]), unpack2x16float(kvs[kb + i + 2u]));
                a3 += dot(unpack2x16float(qs[qb + i + 3u]), unpack2x16float(kvs[kb + i + 3u]));
            }
            let acc = (a0 + a1) + (a2 + a3);
            let vis = visible(qpos[row / hpg], qseq[row / hpg], kpos[key], kseq[key]);
            s[li] = select(NEG, acc, vis);
        }
        workgroupBarrier();
        stage_kv(li, j0, 1u, g);

        // Online softmax: every invocation of a row tracks the same running
        // max m and sum l (same tiles, same order), so the rescale is
        // computed redundantly but consistently; exp(score - m) is taken
        // once per (row, key) by the invocation that scored it.
        var mt = NEG;
        for (var k = 0u; k < KB; k++) { mt = max(mt, s[row * KB + k]); }
        var pr = 0.0;
        if (mt > NEG * 0.5) {
            if (mt > m) {
                let f = exp(m - mt);
                l = l * f;
                for (var i = 0u; i < NV; i++) { o[i] *= f; }
                m = mt;
            }
            let sc = s[li];
            if (sc > NEG * 0.5) { pr = exp(sc - m); }
        }
        ps[li] = pr;
        workgroupBarrier();

        // P.V on this invocation's CH dims of its row.
        if (mt > NEG * 0.5) {
            for (var k = 0u; k < KB; k++) {
                let pk = ps[row * KB + k];
                if (pk > 0.0) {
                    l += pk;
                    let vb = k * HP + chunk / 2u;
                    for (var i = 0u; i < NV; i++) {
                        let a = unpack2x16float(kvs[vb + 2u * i]);
                        let b = unpack2x16float(kvs[vb + 2u * i + 1u]);
                        o[i] += pk * vec4<f32>(a.x, a.y, b.x, b.y);
                    }
                }
            }
        }
    }

    if (ok) {
        let ob = (tok * heads * HD + head * HD + chunk) / 4u;
        let inv = 1.0 / l;
        for (var i = 0u; i < NV; i++) {
            out[ob + i] = o[i] * inv;
        }
    }
}
