// Per token, on the fused projection row
// [q(heads x HD) | k(kv_heads x HD) | v(kv_heads x HD)] (k, v only on layers
// with their own K/V): RMSNorm each q head (q_norm), RoPE the first
// `rope_dims` dims at the token's position with this layer's theta, scale q
// by `scale`, in place. On K/V layers, each k head is normed (k_norm) and
// roped, each v head is RMS-normalized without a weight when `v_norm` is set,
// and both are written to the layer's K/V buffer
// [t][kv_heads x HD | kv_heads x HD] for the attention kernel (and for the
// later layers that share this layer's K/V). One workgroup per token,
// one invocation per HD/256 elements. HD is substituted by the engine.

struct Params {
    t: u32, heads: u32, theta: f32, scale: f32,
    eps: f32, rope_dims: u32, has_kv: u32, v_norm: u32,
    offset: f32, q_stride: u32, kv_heads: u32, _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> qn: array<u32>;
@group(0) @binding(3) var<storage, read> kn: array<u32>;
@group(0) @binding(4) var<storage, read> tok_meta: array<i32>; // (pos, seq) per token
@group(0) @binding(5) var<storage, read_write> kv: array<f32>;

const HD: u32 = 256u;
const PER: u32 = HD / 256u; // elements per invocation

var<workgroup> red: array<f32, 256>;
var<workgroup> tmp: array<f32, HD>;

// Sum of squares over the head row starting at `base`, reduced across the
// workgroup; every invocation returns the total.
fn sumsq(base: u32, li: u32) -> f32 {
    var s = 0.0;
    for (var e = 0u; e < PER; e++) {
        let v = qkv[base + li + 256u * e];
        s += v * v;
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

// rotate_half pairs (d, d + HD/2) for d < rope_dims/2; inv_freq = theta^(-d/(HD/2)).
fn rope_at(d: u32, pos: f32) -> f32 {
    let half = HD / 2u;
    let dd = select(d, d - half, d >= half);
    if (dd >= p.rope_dims / 2u) { return tmp[d]; }
    let angle = pos * pow(p.theta, -f32(dd) / f32(half));
    let c = cos(angle);
    let s = sin(angle);
    if (d < half) {
        return tmp[d] * c - tmp[d + half] * s;
    }
    return tmp[d] * c + tmp[d - half] * s;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let t = wg.x;
    if (t >= p.t) { return; }
    let li = l.x;
    let row = t * p.q_stride;
    let pos = f32(tok_meta[2u * t]);

    // q heads, then the k heads (both normed with a weight and roped).
    let kv_row = t * 2u * p.kv_heads * HD;
    let n_heads = p.heads + p.has_kv * p.kv_heads;
    for (var h = 0u; h < n_heads; h++) {
        let base = row + h * HD;
        let inv = inverseSqrt(sumsq(base, li) / f32(HD) + p.eps);
        for (var e = 0u; e < PER; e++) {
            let d = li + 256u * e;
            var wv: vec2<f32>;
            if (h < p.heads) { wv = unpack2x16float(qn[d / 2u]); } else { wv = unpack2x16float(kn[d / 2u]); }
            let wi = select(wv.x, wv.y, (d & 1u) == 1u);
            tmp[d] = qkv[base + d] * inv * (p.offset + wi);
        }
        workgroupBarrier();
        for (var e = 0u; e < PER; e++) {
            let d = li + 256u * e;
            var y = rope_at(d, pos);
            if (h < p.heads) {
                qkv[base + d] = y * p.scale;
            } else {
                kv[kv_row + (h - p.heads) * HD + d] = y;
            }
        }
        workgroupBarrier();
    }
    if (p.has_kv == 1u) {
        for (var g = 0u; g < p.kv_heads; g++) {
            let base = row + (p.heads + p.kv_heads + g) * HD;
            var inv = 1.0;
            if (p.v_norm == 1u) {
                inv = inverseSqrt(sumsq(base, li) / f32(HD) + p.eps);
            }
            for (var e = 0u; e < PER; e++) {
                let d = li + 256u * e;
                kv[kv_row + (p.kv_heads + g) * HD + d] = qkv[base + d] * inv;
            }
        }
    }
}
