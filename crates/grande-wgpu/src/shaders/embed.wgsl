// out[t, :] = W[ids[t], :] * scale   (Gemma scales embeddings by sqrt(d))
// W is f16, packed two per u32; one invocation writes one pair.

struct Params { t: u32, d: u32, scale: f32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let half_d = p.d / 2u;
    let i = g.x;
    if (i >= p.t * half_d) { return; }
    let row = i / half_d;
    let c2 = i % half_d;
    let v = unpack2x16float(w[ids[row] * half_d + c2]) * p.scale;
    out[2u * i] = v.x;
    out[2u * i + 1u] = v.y;
}
