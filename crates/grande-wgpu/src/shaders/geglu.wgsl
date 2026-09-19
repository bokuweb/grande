// act[t, f] = gelu_tanh(gate[t, f]) * up[t, f], where the fused gate/up
// projection wrote [gate(F) | up(F)] per token.

struct Params { t: u32, f: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gu: array<f32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.t * p.f) { return; }
    let t = i / p.f;
    let f = i % p.f;
    let x = gu[t * 2u * p.f + f];
    let u = gu[t * 2u * p.f + p.f + f];
    // tanh(z) for |z| > ~15 is +-1 to f32 precision; naive GPU tanh
    // implementations overflow to NaN there, so clamp the argument.
    let z = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
    let gelu = 0.5 * x * (1.0 + tanh(z));
    act[i] = gelu * u;
}
