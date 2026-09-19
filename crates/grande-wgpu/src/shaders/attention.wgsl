// Block-causal attention over one packed sequence: prefix tokens (seq 0) then
// branch tokens (seq 1..). A query sees a key iff the key is in the prefix or
// in the query's own branch, is at or before the query's position, and (on
// sliding-window layers) within the window. That is the isolation rule the
// llama.cpp backend gets from its per-sequence KV cache.
//
// One KV head shared by `heads` query heads. Q is read from the layer's fused
// projection buffer (row stride q_stride), K and V from a K/V buffer
// [t][2 x HD] that may belong to an earlier layer (Gemma 4 shares K/V). A
// workgroup of 128 handles 16 query rows = (16 / heads) tokens x heads with
// their Q held as f16 in workgroup memory, and streams keys 8 at a time. A
// tile no row can see is skipped (keys after the block, other branches).
// Otherwise each invocation computes one full (row, key) score, then owns
// HD/8 output dims of one row for the online softmax and P.V. Workgroup
// memory ~12.5 KB at HD 256, ~24.5 KB at 512 (HD is substituted by the engine).

struct Params { t: u32, heads: u32, window: u32, q_stride: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read> tok_meta: array<i32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const HD: u32 = 256u;
const HP: u32 = HD / 2u;   // f16 pairs per head row
const ROWS: u32 = 16u;     // query rows per workgroup = tokens * heads
const KB: u32 = 8u;        // keys per tile
const CH: u32 = HD / 8u;   // output dims per invocation
const NV: u32 = CH / 4u;   // vec4 accumulators per invocation
const NEG: f32 = -1.0e30;

var<workgroup> qs: array<u32, ROWS * HP>;  // [ROWS][HP] f16 pairs
var<workgroup> ks: array<u32, KB * HP>;    // [KB][HP] f16 pairs
var<workgroup> s: array<f32, 128>;         // [ROWS][KB] masked scores
var<workgroup> kpos: array<i32, 8>;
var<workgroup> kseq: array<i32, 8>;
var<workgroup> qpos: array<i32, 16>;
var<workgroup> qseq: array<i32, 16>;
var<workgroup> tile_any: bool;

fn visible(qp: i32, qs_: i32, kp: i32, ks_: i32) -> bool {
    return (ks_ == 0 || ks_ == qs_) && kp <= qp && (p.window == 0u || u32(qp - kp) < p.window);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let heads = p.heads;
    let tb = ROWS / heads;
    let tok0 = wg.x * tb;

    let row = li / KB;          // 0..16, both roles
    let key = li % KB;          // score role
    let chunk = (li % 8u) * CH; // output role: CH dims of `row`
    let tok = tok0 + row / heads;
    let head = row % heads;
    let ok = tok < p.t;

    // Q tile as f16 pairs: CH/2 pairs per invocation.
    {
        let qb = tok * p.q_stride + head * HD + chunk;
        for (var j = 0u; j < CH / 2u; j++) {
            var v = vec2<f32>(0.0);
            if (ok) { v = vec2<f32>(q[qb + 2u * j], q[qb + 2u * j + 1u]); }
            qs[row * HP + chunk / 2u + j] = pack2x16float(v);
        }
    }
    if (li < tb) {
        let tk = tok0 + li;
        if (tk < p.t) {
            qpos[li] = tok_meta[2u * tk];
            qseq[li] = tok_meta[2u * tk + 1u];
        } else {
            qpos[li] = -1;
            qseq[li] = -2;
        }
    }

    var m = NEG;
    var l = 0.0;
    var o: array<vec4<f32>, NV>;
    for (var i = 0u; i < NV; i++) { o[i] = vec4<f32>(0.0); }

    let ntiles = (p.t + KB - 1u) / KB;
    for (var tile = 0u; tile < ntiles; tile++) {
        let j0 = tile * KB;
        if (li < KB) {
            let j = j0 + li;
            if (j < p.t) {
                kpos[li] = tok_meta[2u * j];
                kseq[li] = tok_meta[2u * j + 1u];
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
        // K tile as f16 pairs.
        for (var e = li; e < KB * HP; e += 128u) {
            let kk = e / HP;
            let d2 = e % HP;
            let j = j0 + kk;
            var v = vec2<f32>(0.0);
            if (j < p.t) {
                let b = j * 2u * HD + 2u * d2;
                v = vec2<f32>(kv[b], kv[b + 1u]);
            }
            ks[e] = pack2x16float(v);
        }
        workgroupBarrier();

        // One full score per invocation.
        {
            var acc = 0.0;
            let qb = row * HP;
            let kb = key * HP;
            for (var i = 0u; i < HP; i++) {
                acc += dot(unpack2x16float(qs[qb + i]), unpack2x16float(ks[kb + i]));
            }
            let vis = visible(qpos[row / heads], qseq[row / heads], kpos[key], kseq[key]);
            s[li] = select(NEG, acc, vis);
        }
        workgroupBarrier();

        // Online softmax for this invocation's row, then P.V on its CH dims.
        var mt = NEG;
        for (var k = 0u; k < KB; k++) { mt = max(mt, s[row * KB + k]); }
        if (mt > NEG * 0.5) {
            if (mt > m) {
                let f = exp(m - mt);
                l = l * f;
                for (var i = 0u; i < NV; i++) { o[i] *= f; }
                m = mt;
            }
            for (var k = 0u; k < KB; k++) {
                let sc = s[row * KB + k];
                if (sc > NEG * 0.5) {
                    let pr = exp(sc - m);
                    l += pr;
                    let vb = (j0 + k) * 2u * HD + HD + chunk;
                    for (var i = 0u; i < NV; i++) {
                        let b = vb + 4u * i;
                        o[i] += pr * vec4<f32>(kv[b], kv[b + 1u], kv[b + 2u], kv[b + 3u]);
                    }
                }
            }
        }
    }

    if (ok) {
        let ob = tok * heads * HD + head * HD + chunk;
        let inv = 1.0 / l;
        for (var i = 0u; i < NV; i++) {
            let v = o[i] * inv;
            let b = ob + 4u * i;
            out[b] = v.x;
            out[b + 1u] = v.y;
            out[b + 2u] = v.z;
            out[b + 3u] = v.w;
        }
    }
}
