// out[t, :] (+)= W[ids[t], :] * scale   (Gemma scales embeddings by sqrt(d);
// `residual` adds the row to `out` instead of overwriting it: Laya's type
// embedding onto the encoder output). W is the (possibly quantized)
// embedding table; one invocation per 4 consecutive elements. d is a
// multiple of 32.

struct Params { t: u32, d: u32, scale: f32, residual: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let quads = p.d / 4u;
    let i = g.x;
    if (i >= p.t * quads) { return; }
    let row = i / quads;
    let c = (i % quads) * 4u;
    let e = ids[row] * p.d + c;  // element index into W
    var v: vec4<f32>;
    if (QUANT == 0u) {
        let a = unpack2x16float(w[e / 2u]);
        let b = unpack2x16float(w[e / 2u + 1u]);
        v = vec4<f32>(a, b);
    } else {
        let blk = e / 32u;
        let d = block_scale(sc[blk / 2u], blk);
        if (QUANT == 1u) {
            v = dq8(w[e / 4u], d);
        } else {
            let j = e % 32u;
            let word = w[blk * 4u + (j % 16u) / 4u];
            if (j < 16u) { v = dq4lo(word, d); } else { v = dq4hi(word, d); }
        }
    }
    v = v * p.scale;
    let o = row * p.d + c;
    if (p.residual == 1u) {
        v += vec4<f32>(out[o], out[o + 1u], out[o + 2u], out[o + 3u]);
    }
    out[o] = v.x;
    out[o + 1u] = v.y;
    out[o + 2u] = v.z;
    out[o + 3u] = v.w;
}
