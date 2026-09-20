// LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b, one workgroup per
// row (ModernBERT / Laya's decision head). `w` and `b` are f16 pairs; b is
// read only when has_bias is set (a dummy buffer is bound otherwise).
// Variance is taken around the mean (two reductions), not as E[x^2] - m^2.

struct Params { t: u32, d: u32, eps: f32, has_bias: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> b: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

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

fn half_at(buf_w: bool, i: u32) -> f32 {
    var v: vec2<f32>;
    if (buf_w) { v = unpack2x16float(w[i / 2u]); } else { v = unpack2x16float(b[i / 2u]); }
    return select(v.x, v.y, (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let row = wg.x;
    if (row >= p.t) { return; }
    let base = row * p.d;
    var s = 0.0;
    for (var i = l.x; i < p.d; i += 256u) { s += x[base + i]; }
    let mean = reduce(s, l.x) / f32(p.d);
    var q = 0.0;
    for (var i = l.x; i < p.d; i += 256u) {
        let c = x[base + i] - mean;
        q += c * c;
    }
    let inv = inverseSqrt(reduce(q, l.x) / f32(p.d) + p.eps);
    for (var i = l.x; i < p.d; i += 256u) {
        var y = (x[base + i] - mean) * inv * half_at(true, i);
        if (p.has_bias == 1u) { y += half_at(false, i); }
        out[base + i] = y;
    }
}
