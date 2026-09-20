// RoPE for a bidirectional encoder (ModernBERT): rotate every q and k head
// of the fused [q | k | v] projection row in place at the token's position
// (rotate_half pairs (d, d + HD/2), inv_freq theta^(-2d/HD), the whole
// head), and scale q by `scale` (HD^-0.5). One invocation per (token, head,
// pair). HD is substituted by the engine.

struct Params { t: u32, heads: u32, stride: u32, k_off: u32, theta: f32, scale: f32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tok_meta: array<i32>; // (pos, seq, lo, hi) per token
@group(0) @binding(2) var<storage, read_write> qkv: array<f32>;

const HD: u32 = 256u;
const HALF: u32 = HD / 2u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let per_tok = p.heads * HALF;
    if (i >= p.t * per_tok) { return; }
    let tok = i / per_tok;
    let h = (i % per_tok) / HALF;
    let d = i % HALF;
    let pos = f32(tok_meta[4u * tok]);
    let angle = pos * pow(p.theta, -f32(2u * d) / f32(HD));
    let c = cos(angle);
    let s = sin(angle);
    let row = tok * p.stride + h * HD;
    // q
    {
        let a = qkv[row + d];
        let b = qkv[row + d + HALF];
        qkv[row + d] = (a * c - b * s) * p.scale;
        qkv[row + d + HALF] = (b * c + a * s) * p.scale;
    }
    // k
    {
        let kr = row + p.k_off;
        let a = qkv[kr + d];
        let b = qkv[kr + d + HALF];
        qkv[kr + d] = a * c - b * s;
        qkv[kr + d + HALF] = b * c + a * s;
    }
}
