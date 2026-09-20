// Bidirectional attention over one packed batch of sequences (ModernBERT /
// Laya): a query sees a key iff both are in the same sequence and, on
// local layers, |query position - key position| <= window. Q, K and V are
// read from the fused projection buffer [t][q | k | v] (row stride
// `stride`, K at `k_off`, V at `v_off`, `heads` heads of HD each; no K/V
// sharing, no cache). Otherwise the same tiling as attention.wgsl: a
// workgroup of ROWS x KB invocations handles ROWS query rows of one head
// (wg.y) and streams that head's keys KB at a time, restricted to the cache
// rows the tokens' sequences span (tok_meta: position, sequence, first row,
// last row + 1). HD, ROWS and KB are substituted by the engine.

struct Params { t: u32, heads: u32, window: u32, stride: u32, k_off: u32, v_off: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tok_meta: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

const HD: u32 = 256u;
const ROWS: u32 = 16u;     // query rows (tokens) per workgroup
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

fn visible(qp: i32, qs_: i32, kp: i32, ks_: i32) -> bool {
    return ks_ == qs_ && (p.window == 0u || u32(abs(qp - kp)) <= p.window);
}

// Stage KB rows of K (half == 0) or V (half == 1) of head g, tile j0, as f16 pairs.
fn stage_kv(li: u32, j0: u32, half: u32, g: u32) {
    let stride4 = p.stride / 4u;
    let off4 = select(p.k_off, p.v_off, half == 1u) / 4u + g * HQ;
    for (var e = li; e < KB * HQ; e += N) {
        let kk = e / HQ;
        let d4 = e % HQ;
        let j = j0 + kk;
        var v = vec4<f32>(0.0);
        if (j < p.t) { v = qkv[j * stride4 + off4 + d4]; }
        kvs[kk * HP + 2u * d4] = pack2x16float(v.xy);
        kvs[kk * HP + 2u * d4 + 1u] = pack2x16float(v.zw);
    }
}

@compute @workgroup_size(128) // ROWS * KB, substituted by the engine
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let g = wg.y;                 // this workgroup's head
    let tok0 = wg.x * ROWS;

    let row = li / KB;          // 0..ROWS, both roles
    let key = li % KB;          // score role
    let chunk = key * CH;       // output role: CH dims of `row`
    let tok = tok0 + row;
    let ok = tok < p.t;

    // Q tile as f16 pairs: this invocation's CH dims of its row.
    {
        let qb = (tok * p.stride + g * HD + chunk) / 4u;
        for (var i = 0u; i < NV; i++) {
            var v = vec4<f32>(0.0);
            if (ok) { v = qkv[qb + i]; }
            qs[row * HP + chunk / 2u + 2u * i] = pack2x16float(v.xy);
            qs[row * HP + chunk / 2u + 2u * i + 1u] = pack2x16float(v.zw);
        }
    }
    if (li < ROWS) {
        let tk = tok0 + li;
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
    // Keys these tokens can see lie in rows [lo, hi): their sequences' spans.
    if (li == 0u) {
        var lo = 0x7fffffff;
        var hi = 0;
        for (var qi = 0u; qi < ROWS; qi++) {
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
        if (li == 0u) {
            var any = false;
            for (var k = 0u; k < KB; k++) {
                for (var qi = 0u; qi < ROWS; qi++) {
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
            let vis = visible(qpos[row], qseq[row], kpos[key], kseq[key]);
            s[li] = select(NEG, acc, vis);
        }
        workgroupBarrier();
        stage_kv(li, j0, 1u, g);

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
        let ob = (tok * p.heads * HD + g * HD + chunk) / 4u;
        let inv = 1.0 / l;
        for (var i = 0u; i < NV; i++) {
            out[ob + i] = o[i] * inv;
        }
    }
}
