// Per token, in place on the fused [q(heads x 256) | k(256) | v(256)] row:
// RMSNorm each q head and k over head_dim (q_norm / k_norm), then RoPE with
// this layer's theta at the token's position, then scale q by
// query_pre_attn_scalar^-0.5. One workgroup per token, one invocation per
// head_dim element.

struct Params { t: u32, heads: u32, theta: f32, scale: f32, eps: f32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> qn: array<u32>;
@group(0) @binding(3) var<storage, read> kn: array<u32>;
@group(0) @binding(4) var<storage, read> tok_meta: array<i32>; // (pos, seq) per token

const HD: u32 = 256u;

var<workgroup> red: array<f32, 256>;
var<workgroup> tmp: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let t = wg.x;
    if (t >= p.t) { return; }
    let d = l.x;
    let row = t * (p.heads + 2u) * HD;
    let pos = f32(tok_meta[2u * t]);
    // rotate_half pairs (d, d + 128); inv_freq = theta^(-d/128)
    let half = HD / 2u;
    let dd = select(d, d - half, d >= half);
    let angle = pos * pow(p.theta, -f32(dd) / f32(half));
    let c = cos(angle);
    let s = sin(angle);

    for (var h = 0u; h <= p.heads; h++) {
        let base = row + h * HD;
        let v = qkv[base + d];
        red[d] = v * v;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st >>= 1u) {
            if (d < st) { red[d] += red[d + st]; }
            workgroupBarrier();
        }
        let inv = inverseSqrt(red[0] / f32(HD) + p.eps);
        var wv: vec2<f32>;
        if (h < p.heads) { wv = unpack2x16float(qn[d / 2u]); } else { wv = unpack2x16float(kn[d / 2u]); }
        let wi = select(wv.x, wv.y, (d & 1u) == 1u);
        tmp[d] = v * inv * (1.0 + wi);
        workgroupBarrier();
        var y: f32;
        if (d < half) {
            y = tmp[d] * c - tmp[d + half] * s;
        } else {
            y = tmp[d] * c + tmp[d - half] * s;
        }
        if (h < p.heads) { y = y * p.scale; }
        qkv[base + d] = y;
        workgroupBarrier();
    }
}
