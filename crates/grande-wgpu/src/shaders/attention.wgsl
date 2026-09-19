// Block-causal attention over one packed sequence: prefix tokens (seq 0) then
// branch tokens (seq 1..). A query sees a key iff the key is in the prefix or
// in the query's own branch, is at or before the query's position, and (on
// sliding-window layers) within the window. That is the isolation rule the
// llama.cpp backend gets from its per-sequence KV cache.
//
// One KV head shared by `heads` query heads (Gemma 3 270M: 4 : 1). A workgroup
// of 128 handles 16 query rows = (16 / heads) tokens x heads with their Q held
// as f16 in workgroup memory, and streams keys 8 at a time. A tile no row can
// see is skipped (keys after the block, other branches). Otherwise each
// invocation computes one full (row, key) score, then owns 32 output dims of
// one row for the online softmax and P.V (V read straight from the qkv
// buffer). Workgroup memory ~12.5 KB, so several workgroups share a core.

struct Params { t: u32, heads: u32, window: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> tok_meta: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const HD: u32 = 256u;
const HP: u32 = 128u;  // f16 pairs per head row
const ROWS: u32 = 16u; // query rows per workgroup = tokens * heads
const KB: u32 = 8u;    // keys per tile
const NEG: f32 = -1.0e30;

var<workgroup> qs: array<u32, 2048>;  // [ROWS][HP] f16 pairs
var<workgroup> ks: array<u32, 1024>;  // [KB][HP] f16 pairs
var<workgroup> s: array<f32, 128>;    // [ROWS][KB] masked scores
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
    let stride = (heads + 2u) * HD;
    let tb = ROWS / heads;
    let tok0 = wg.x * tb;

    let row = li / KB;          // 0..16, both roles
    let key = li % KB;          // score role
    let chunk = (li % 8u) * 32u; // output role: 32 dims of `row`
    let tok = tok0 + row / heads;
    let head = row % heads;
    let ok = tok < p.t;

    // Q tile as f16 pairs: 16 pairs per invocation.
    {
        let qb = tok * stride + head * HD + chunk;
        for (var j = 0u; j < 16u; j++) {
            var v = vec2<f32>(0.0);
            if (ok) { v = vec2<f32>(qkv[qb + 2u * j], qkv[qb + 2u * j + 1u]); }
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
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    var o4 = vec4<f32>(0.0);
    var o5 = vec4<f32>(0.0);
    var o6 = vec4<f32>(0.0);
    var o7 = vec4<f32>(0.0);

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
                for (var q = 0u; q < tb; q++) {
                    any = any || visible(qpos[q], qseq[q], kpos[k], kseq[k]);
                }
            }
            tile_any = any;
        }
        if (!workgroupUniformLoad(&tile_any)) {
            continue;
        }
        // K tile as f16 pairs: 8 pairs per invocation.
        for (var e = li; e < KB * HP; e += 128u) {
            let kk = e / HP;
            let d2 = e % HP;
            let j = j0 + kk;
            var v = vec2<f32>(0.0);
            if (j < p.t) {
                let b = j * stride + heads * HD + 2u * d2;
                v = vec2<f32>(qkv[b], qkv[b + 1u]);
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

        // Online softmax for this invocation's row, then P.V on its 32 dims.
        var mt = NEG;
        for (var k = 0u; k < KB; k++) { mt = max(mt, s[row * KB + k]); }
        if (mt > NEG * 0.5) {
            if (mt > m) {
                let f = exp(m - mt);
                l = l * f;
                o0 *= f; o1 *= f; o2 *= f; o3 *= f; o4 *= f; o5 *= f; o6 *= f; o7 *= f;
                m = mt;
            }
            for (var k = 0u; k < KB; k++) {
                let sc = s[row * KB + k];
                if (sc > NEG * 0.5) {
                    let pr = exp(sc - m);
                    l += pr;
                    let vb = (j0 + k) * stride + (heads + 1u) * HD + chunk;
                    o0 += pr * vec4<f32>(qkv[vb], qkv[vb + 1u], qkv[vb + 2u], qkv[vb + 3u]);
                    o1 += pr * vec4<f32>(qkv[vb + 4u], qkv[vb + 5u], qkv[vb + 6u], qkv[vb + 7u]);
                    o2 += pr * vec4<f32>(qkv[vb + 8u], qkv[vb + 9u], qkv[vb + 10u], qkv[vb + 11u]);
                    o3 += pr * vec4<f32>(qkv[vb + 12u], qkv[vb + 13u], qkv[vb + 14u], qkv[vb + 15u]);
                    o4 += pr * vec4<f32>(qkv[vb + 16u], qkv[vb + 17u], qkv[vb + 18u], qkv[vb + 19u]);
                    o5 += pr * vec4<f32>(qkv[vb + 20u], qkv[vb + 21u], qkv[vb + 22u], qkv[vb + 23u]);
                    o6 += pr * vec4<f32>(qkv[vb + 24u], qkv[vb + 25u], qkv[vb + 26u], qkv[vb + 27u]);
                    o7 += pr * vec4<f32>(qkv[vb + 28u], qkv[vb + 29u], qkv[vb + 30u], qkv[vb + 31u]);
                }
            }
        }
    }

    if (ok) {
        let ob = tok * heads * HD + head * HD + chunk;
        let inv = 1.0 / l;
        store4(ob, o0 * inv);
        store4(ob + 4u, o1 * inv);
        store4(ob + 8u, o2 * inv);
        store4(ob + 12u, o3 * inv);
        store4(ob + 16u, o4 * inv);
        store4(ob + 20u, o5 * inv);
        store4(ob + 24u, o6 * inv);
        store4(ob + 28u, o7 * inv);
    }
}

fn store4(b: u32, v: vec4<f32>) {
    out[b] = v.x;
    out[b + 1u] = v.y;
    out[b + 2u] = v.z;
    out[b + 3u] = v.w;
}
