// LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b, one workgroup per
// row (ModernBERT / Laya's decision head). With `residual` the row is
// first updated in place from a linear layer's output, x += a (+ rb, that
// layer's bias), so a residual add and the norm that follows it are one
// dispatch; `in_place` writes the normed row over x instead of into `out`
// (the last norm of the residual stream, whose sum nobody reads). `w`, `b`
// and `rb` are f16 pairs; b / rb are read only when has_bias / has_rbias
// is set (dummy buffers are bound otherwise). Variance is taken around
// the mean (two reductions), not as E[x^2] - m^2. Each invocation keeps
// its elements in registers between the passes: d <= 2048.

struct Params { t: u32, d: u32, eps: f32, has_bias: u32, residual: u32, has_rbias: u32, in_place: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> b: array<u32>;
@group(0) @binding(4) var<storage, read> rb: array<u32>;
@group(0) @binding(5) var<storage, read_write> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

const PER: u32 = 8u;  // elements per invocation: d <= 256 * PER

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

// Element i of one of the f16-pair vectors: 0 = w, 1 = b, 2 = rb.
fn half_at(which: u32, i: u32) -> f32 {
    var v: vec2<f32>;
    if (which == 0u) {
        v = unpack2x16float(w[i / 2u]);
    } else if (which == 1u) {
        v = unpack2x16float(b[i / 2u]);
    } else {
        v = unpack2x16float(rb[i / 2u]);
    }
    return select(v.x, v.y, (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let row = wg.x;
    if (row >= p.t) { return; }
    let base = row * p.d;
    var v: array<f32, PER>;
    var s = 0.0;
    for (var k = 0u; k < PER; k++) {
        let i = l.x + 256u * k;
        var e = 0.0;
        if (i < p.d) {
            e = x[base + i];
            if (p.residual == 1u) {
                e += a[base + i];
                if (p.has_rbias == 1u) { e += half_at(2u, i); }
                x[base + i] = e;
            }
        }
        v[k] = e;
        s += e;
    }
    let mean = reduce(s, l.x) / f32(p.d);
    var q = 0.0;
    for (var k = 0u; k < PER; k++) {
        let i = l.x + 256u * k;
        if (i < p.d) {
            let c = v[k] - mean;
            q += c * c;
        }
    }
    let inv = inverseSqrt(reduce(q, l.x) / f32(p.d) + p.eps);
    for (var k = 0u; k < PER; k++) {
        let i = l.x + 256u * k;
        if (i < p.d) {
            var y = (v[k] - mean) * inv * half_at(0u, i);
            if (p.has_bias == 1u) { y += half_at(1u, i); }
            if (p.in_place == 1u) { x[base + i] = y; } else { out[base + i] = y; }
        }
    }
}
