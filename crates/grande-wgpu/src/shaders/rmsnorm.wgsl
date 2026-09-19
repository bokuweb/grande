// Gemma RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w), one workgroup per row.
// residual = 1 adds the result into `out` (the post-attention / post-feedforward
// norms feed the residual stream), otherwise `out` is overwritten.

struct Params { t: u32, d: u32, eps: f32, residual: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

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
    red[l.x] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (l.x < st) { red[l.x] += red[l.x + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(p.d) + p.eps);
    for (var i = l.x; i < p.d; i += 256u) {
        let wv = unpack2x16float(w[i / 2u]);
        let wi = select(wv.x, wv.y, (i & 1u) == 1u);
        let y = x[base + i] * inv * (1.0 + wi);
        if (p.residual == 1u) {
            out[base + i] = out[base + i] + y;
        } else {
            out[base + i] = y;
        }
    }
}
