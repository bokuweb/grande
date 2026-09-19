// Gemma 4 per-layer inputs, once per request before the layers:
//   pli[t, l, :] = (RMSNorm(proj[t, l, :] * proj_scale; w) + emb[t, l, :] * emb_scale) * out_scale
// proj is x . per_layer_model_proj^T laid out [t][layers x P]; emb is the
// gathered per-layer token table (f16 pairs, same layout, uploaded by the
// host); proj_scale = 1/sqrt(d), emb_scale = sqrt(P), out_scale = 1/sqrt(2).
// One workgroup per (token, layer) slice of P elements.

struct Params { t: u32, p: u32, layers: u32, eps: f32, offset: f32, proj_scale: f32, emb_scale: f32, out_scale: f32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> emb: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let slice = wg.x; // t * layers + layer
    if (slice >= p.t * p.layers) { return; }
    let base = slice * p.p;
    var s = 0.0;
    for (var i = l.x; i < p.p; i += 256u) {
        let v = proj[base + i] * p.proj_scale;
        s += v * v;
    }
    red[l.x] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (l.x < st) { red[l.x] += red[l.x + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(p.p) + p.eps);
    for (var i = l.x; i < p.p; i += 256u) {
        let wv = unpack2x16float(w[i / 2u]);
        let wi = select(wv.x, wv.y, (i & 1u) == 1u);
        let n = proj[base + i] * p.proj_scale * inv * (p.offset + wi);
        let ev = unpack2x16float(emb[(base + i) / 2u]);
        let e = select(ev.x, ev.y, ((base + i) & 1u) == 1u);
        out[base + i] = (n + e * p.emb_scale) * p.out_scale;
    }
}
