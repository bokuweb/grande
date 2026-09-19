// Y[M, N] = X[M, K] * W[N, K]^T for a handful of rows and a huge N: the
// label readout's logits, M = wanted positions (typically < 16), N = vocab
// (262144), W the (quantized) embedding table. The tiled matmul pads M to 64
// and wastes most of its work here; this kernel gives every invocation one
// column n and streams K in 128-wide slices with the X rows staged in
// workgroup memory, 16 rows per pass. K must be a multiple of 32.

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const RM: u32 = 16u;  // rows per pass
const KS: u32 = 128u; // k per slice

var<workgroup> xs: array<vec4<f32>, RM * KS / 4u>; // [RM][KS/4]

// Elements kk..kk+4 of column n (kk a multiple of 4).
fn wquad(n: u32, kk: u32) -> vec4<f32> {
    let e = n * p.k + kk;
    if (QUANT == 0u) {
        return vec4<f32>(unpack2x16float(w[e / 2u]), unpack2x16float(w[e / 2u + 1u]));
    }
    let blk = e / 32u;
    let d = block_scale(sc[blk / 2u], blk);
    if (QUANT == 1u) {
        return dq8(w[e / 4u], d);
    }
    let j = e % 32u;
    let q = w[blk * 4u + (j % 16u) / 4u];
    if (j < 16u) { return dq4lo(q, d); }
    return dq4hi(q, d);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = wg.x * 256u + li;
    let ok = n < p.n;
    for (var r0 = 0u; r0 < p.m; r0 += RM) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var k0 = 0u; k0 < p.k; k0 += KS) {
            // Stage X[r0..r0+16][k0..k0+128]: 512 vec4, two per invocation.
            for (var i = li; i < RM * KS / 4u; i += 256u) {
                let r = i / (KS / 4u);
                let c = (i % (KS / 4u)) * 4u;
                var v = vec4<f32>(0.0);
                if (r0 + r < p.m && k0 + c < p.k) {
                    let b = (r0 + r) * p.k + k0 + c;
                    v = vec4<f32>(x[b], x[b + 1u], x[b + 2u], x[b + 3u]);
                }
                xs[i] = v;
            }
            workgroupBarrier();
            if (ok) {
                let nq = min(KS, p.k - k0) / 4u;
                for (var q = 0u; q < nq; q++) {
                    let wv = wquad(n, k0 + 4u * q);
                    a0 += vec4<f32>(dot(xs[q], wv), dot(xs[32u + q], wv), dot(xs[64u + q], wv), dot(xs[96u + q], wv));
                    a1 += vec4<f32>(dot(xs[128u + q], wv), dot(xs[160u + q], wv), dot(xs[192u + q], wv), dot(xs[224u + q], wv));
                    a2 += vec4<f32>(dot(xs[256u + q], wv), dot(xs[288u + q], wv), dot(xs[320u + q], wv), dot(xs[352u + q], wv));
                    a3 += vec4<f32>(dot(xs[384u + q], wv), dot(xs[416u + q], wv), dot(xs[448u + q], wv), dot(xs[480u + q], wv));
                }
            }
            workgroupBarrier();
        }
        if (ok) {
            store(r0, n, a0);
            store(r0 + 4u, n, a1);
            store(r0 + 8u, n, a2);
            store(r0 + 12u, n, a3);
        }
    }
}

fn store(r: u32, n: u32, v: vec4<f32>) {
    if (r < p.m) { y[r * p.n + n] = v.x; }
    if (r + 1u < p.m) { y[(r + 1u) * p.n + n] = v.y; }
    if (r + 2u < p.m) { y[(r + 2u) * p.n + n] = v.z; }
    if (r + 3u < p.m) { y[(r + 3u) * p.n + n] = v.w; }
}
