// Block-causal attention over one packed sequence: prefix tokens (seq 0) then
// branch tokens (seq 1..). A query sees a key iff the key is in the prefix or
// in the query's own branch, is at or before the query's position, and (on
// sliding-window layers) within the window. That is the isolation rule the
// llama.cpp backend gets from its per-sequence KV cache.
//
// One KV head shared by `heads` query heads (Gemma 3 270M: 4 : 1). A workgroup
// of 256 handles 16 query rows = (16 / heads) tokens x heads and streams keys
// through workgroup memory 8 at a time: scores by 2 invocations per (row,
// key), then an online softmax where each invocation owns 16 output dims of
// one row.

struct Params { t: u32, heads: u32, window: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> tok_meta: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const HD: u32 = 256u;
const ROWS: u32 = 16u; // query rows per workgroup = tokens * heads
const KB: u32 = 8u;    // keys per tile
const NEG: f32 = -1.0e30;

var<workgroup> kt: array<f32, 2048>;   // [HD][KB]  (dim-major)
var<workgroup> vt: array<u32, 1024>;   // [KB][HD/2] f16 pairs
var<workgroup> part: array<f32, 256>;  // score partials
var<workgroup> s: array<f32, 128>;     // [ROWS][KB] masked scores
var<workgroup> kpos: array<i32, 8>;
var<workgroup> kseq: array<i32, 8>;
var<workgroup> qpos: array<i32, 16>;
var<workgroup> qseq: array<i32, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let heads = p.heads;
    let stride = (heads + 2u) * HD;
    let tb = ROWS / heads; // tokens per workgroup
    let tok0 = wg.x * tb;

    // Score role: pair = (row, key), two invocations per pair (dim halves).
    let pair = li >> 1u;
    let half = li & 1u;
    let srow = pair / KB;
    let skey = pair % KB;
    let stok = tok0 + srow / heads;
    let shead = srow % heads;
    let s_ok = stok < p.t;
    let qbase = stok * stride + shead * HD + half * 128u;

    // Output role: row = li / 16, dims chunk = (li % 16) * 16.
    let orow = li / 16u;
    let chunk = (li % 16u) * 16u;
    let otok = tok0 + orow / heads;
    let ohead = orow % heads;
    let o_ok = otok < p.t;

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
    var o: array<f32, 16>;
    for (var j = 0u; j < 16u; j++) { o[j] = 0.0; }

    let ntiles = (p.t + KB - 1u) / KB;
    for (var tile = 0u; tile < ntiles; tile++) {
        let j0 = tile * KB;
        // K tile, dim-major: 8 values per invocation.
        for (var e = li; e < KB * HD; e += 256u) {
            let key = e / HD;
            let d = e % HD;
            let j = j0 + key;
            var v = 0.0;
            if (j < p.t) { v = qkv[j * stride + heads * HD + d]; }
            kt[d * KB + key] = v;
        }
        // V tile as f16 pairs: 4 per invocation.
        for (var e = li; e < KB * HD / 2u; e += 256u) {
            let key = e / (HD / 2u);
            let d2 = e % (HD / 2u);
            let j = j0 + key;
            var v = vec2<f32>(0.0);
            if (j < p.t) {
                let b = j * stride + (heads + 1u) * HD + 2u * d2;
                v = vec2<f32>(qkv[b], qkv[b + 1u]);
            }
            vt[e] = pack2x16float(v);
        }
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

        // Scores: half a dot product per invocation.
        var acc = 0.0;
        if (s_ok) {
            let kb = half * 128u * KB + skey;
            for (var i = 0u; i < 128u; i++) {
                acc += qkv[qbase + i] * kt[kb + i * KB];
            }
        }
        part[li] = acc;
        workgroupBarrier();
        if (li < ROWS * KB) {
            let row = li / KB;
            let key = li % KB;
            let tk = row / heads;
            let qp = qpos[tk];
            let qs = qseq[tk];
            let kp = kpos[key];
            let ks = kseq[key];
            let visible = (ks == 0 || ks == qs) && kp <= qp
                && (p.window == 0u || u32(qp - kp) < p.window);
            s[li] = select(NEG, part[2u * li] + part[2u * li + 1u], visible);
        }
        workgroupBarrier();

        // Online softmax for this invocation's row, then P.V on its 16 dims.
        var mt = NEG;
        for (var k = 0u; k < KB; k++) { mt = max(mt, s[orow * KB + k]); }
        if (mt > NEG * 0.5) {
            if (mt > m) {
                let f = exp(m - mt);
                l = l * f;
                for (var j = 0u; j < 16u; j++) { o[j] = o[j] * f; }
                m = mt;
            }
            for (var k = 0u; k < KB; k++) {
                let sc = s[orow * KB + k];
                if (sc > NEG * 0.5) {
                    let pr = exp(sc - m);
                    l += pr;
                    let vb = k * (HD / 2u) + chunk / 2u;
                    for (var j2 = 0u; j2 < 8u; j2++) {
                        let v = unpack2x16float(vt[vb + j2]);
                        o[2u * j2] += pr * v.x;
                        o[2u * j2 + 1u] += pr * v.y;
                    }
                }
            }
        }
        workgroupBarrier();
    }

    if (o_ok) {
        let ob = otok * heads * HD + ohead * HD + chunk;
        let inv = 1.0 / l;
        for (var j = 0u; j < 16u; j++) { out[ob + j] = o[j] * inv; }
    }
}
