// Gemma RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (offset + w), one workgroup
// per row. offset is 1 for Gemma 3 (weights stored as w, applied as 1 + w)
// and 0 for Gemma 4 / GGUF-derived weights. residual = 1 adds the result
// into `out` (the post-attention / post-feedforward norms feed the residual
// stream), otherwise `out` is overwritten; the row is then multiplied by
// `scale` (Gemma 4's per-layer output scalar, 1 elsewhere). second = 1
// then also norms the row just written with `w2` into `out2`: the residual
// add and the norm that reads the stream next (pre-feedforward, or the next
// layer's input norm) are one dispatch. w2 / out2 are dummies otherwise.

struct Params { t: u32, d: u32, eps: f32, residual: u32, offset: f32, scale: f32, second: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> w2: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> out2: array<f32>;

var<workgroup> red: array<f32, 256>;

fn reduce(v: f32, li: u32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

// Element i of w (which == 0) or w2, f16 pairs.
fn weight(which: u32, i: u32) -> f32 {
    var v: vec2<f32>;
    if (which == 0u) { v = unpack2x16float(w[i / 2u]); } else { v = unpack2x16float(w2[i / 2u]); }
    return p.offset + select(v.x, v.y, (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let row = wg.x;
    if (row >= p.t) { return; }
    let base = row * p.d;
    var s = 0.0;
    for (var i = l.x; i < p.d; i += 256u) {
        let v = x[base + i];
        s += v * v;
    }
    let inv = inverseSqrt(reduce(s, l.x) / f32(p.d) + p.eps);
    // The same invocation writes and re-reads each of its elements, so the
    // second pass needs no barrier.
    var s2 = 0.0;
    for (var i = l.x; i < p.d; i += 256u) {
        var y = x[base + i] * inv * weight(0u, i);
        if (p.residual == 1u) { y += out[base + i]; }
        y *= p.scale;
        out[base + i] = y;
        s2 += y * y;
    }
    if (p.second == 1u) {
        let inv2 = inverseSqrt(reduce(s2, l.x) / f32(p.d) + p.eps);
        for (var i = l.x; i < p.d; i += 256u) {
            out2[base + i] = out[base + i] * inv2 * weight(1u, i);
        }
    }
}
