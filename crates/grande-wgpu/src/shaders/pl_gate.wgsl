// Gemma 4 per-layer gate: g[t, :] = gelu_tanh(g[t, :]) * pli[t, layer, :],
// in place on the P-wide gate projection. One invocation per element.

struct Params { t: u32, p: u32, layers: u32, layer: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pli: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;

fn gelu(x: f32) -> f32 {
    let z = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(z));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.t * p.p) { return; }
    let t = i / p.p;
    let c = i % p.p;
    g[i] = gelu(g[i]) * pli[(t * p.layers + p.layer) * p.p + c];
}
