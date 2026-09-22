// The (state, option) head's input rows: out[i, :] = [s, o, |s - o|, s * o]
// for pair i = (state slot, option slot) over unit vectors in `vectors`
// (e5.rs). One invocation per (pair, column).

struct Params { n: u32, d: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.n * p.d) { return; }
    let pr = i / p.d;
    let c = i % p.d;
    let s = vectors[pairs[2u * pr] * p.d + c];
    let o = vectors[pairs[2u * pr + 1u] * p.d + c];
    let base = pr * 4u * p.d + c;
    out[base] = s;
    out[base + p.d] = o;
    out[base + 2u * p.d] = abs(s - o);
    out[base + 3u * p.d] = s * o;
}
