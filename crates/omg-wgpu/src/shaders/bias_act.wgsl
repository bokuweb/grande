// Elementwise tail of a linear layer over a [t, n] activation:
//   y = act(src + bias);  out = y (+ out when residual)
// where src is `a` (a matmul's output) or, for an in-place bias, `out`
// itself (`in_place`). wgpu forbids binding one buffer both read-only and
// read-write in a dispatch, so the residual stream is always the
// read-write `out`. `bias` (f16 pairs, per column) is read when has_bias
// is set; a dummy buffer is bound otherwise. mode: 0 = identity, 1 = ReLU,
// 2 = exact (erf) GELU (BERT's FFN, e5.rs).

struct Params { t: u32, n: u32, mode: u32, has_bias: u32, in_place: u32, residual: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// erf to 1.5e-7 (Abramowitz & Stegun 7.1.26), as matmul_gated.wgsl.
fn erf1(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-a * a));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.t * p.n) { return; }
    var y = select(a[i], out[i], p.in_place == 1u);
    if (p.has_bias == 1u) {
        let c = i % p.n;
        let v = unpack2x16float(bias[c / 2u]);
        y += select(v.x, v.y, (c & 1u) == 1u);
    }
    if (p.mode == 1u) { y = max(y, 0.0); }
    if (p.mode == 2u) { y = 0.5 * y * (1.0 + erf1(y * 0.7071067811865476)); }
    if (p.residual == 1u) { y += out[i]; }
    out[i] = y;
}
