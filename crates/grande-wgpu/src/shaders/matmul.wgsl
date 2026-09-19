// Y[M, N] = X[M, K] * W[N, K]^T   (a linear layer; W row-major as HF stores it)
// X, Y are f32; W is f16 packed two per u32 or a quantized block format (see
// weight.wgsl, prepended; QUANT picks the decoder). A 64x64 output tile per
// workgroup of 64 invocations, each owning 8 consecutive rows x 8 consecutive
// cols. K streams through workgroup memory 16 wide as k-major vec4 tiles, so
// the inner loop is 4 vector loads per 64 FMAs and a simdgroup's loads are
// contiguous. K must be a multiple of 32; M and N are bounds-checked.

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const G: u32 = 16u; // vec4 groups per tile row (BM / 4 = BN / 4)

var<workgroup> xs: array<vec4<f32>, 256>; // [BK][G]: rows 4g..4g+4 at k
var<workgroup> ws: array<vec4<f32>, 256>; // [BK][G]: cols 4g..4g+4 at k

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    let tr = li / 8u; // rows tr*8 .. tr*8+8
    let tc = li % 8u; // cols tc*8 .. tc*8+8
    var lo0 = vec4<f32>(0.0); var hi0 = vec4<f32>(0.0);
    var lo1 = vec4<f32>(0.0); var hi1 = vec4<f32>(0.0);
    var lo2 = vec4<f32>(0.0); var hi2 = vec4<f32>(0.0);
    var lo3 = vec4<f32>(0.0); var hi3 = vec4<f32>(0.0);
    var lo4 = vec4<f32>(0.0); var hi4 = vec4<f32>(0.0);
    var lo5 = vec4<f32>(0.0); var hi5 = vec4<f32>(0.0);
    var lo6 = vec4<f32>(0.0); var hi6 = vec4<f32>(0.0);
    var lo7 = vec4<f32>(0.0); var hi7 = vec4<f32>(0.0);

    let ktiles = p.k / BK;
    for (var kt = 0u; kt < ktiles; kt++) {
        let k0 = kt * BK;
        // X tile: invocation li reads k = li % 16 of rows 4g..4g+4 for
        // g = li/16 + 4i (16 consecutive k of a row across 16 invocations).
        {
            let c = li % BK;
            for (var i = 0u; i < 4u; i++) {
                let g = li / BK + 4u * i;
                var v = vec4<f32>(0.0);
                for (var j = 0u; j < 4u; j++) {
                    let gr = row0 + 4u * g + j;
                    if (gr < p.m) { v[j] = x[gr * p.k + k0 + c]; }
                }
                xs[c * G + g] = v;
            }
        }
        // W tile. f16: invocation li reads k pair li % 8 of cols 4g..4g+4
        // for g = li/8 + 8i. Quantized: invocation li decodes k quad
        // (li % 4) * 4 of cols 4g..4g+4 for g = li / 4.
        if (QUANT == 0u) {
            let c2 = li % 8u;
            for (var i = 0u; i < 2u; i++) {
                let g = li / 8u + 8u * i;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                for (var j = 0u; j < 4u; j++) {
                    let gn = col0 + 4u * g + j;
                    if (gn < p.n) {
                        let v = unpack2x16float(w[(gn * p.k + k0) / 2u + c2]);
                        a[j] = v.x;
                        b[j] = v.y;
                    }
                }
                ws[(2u * c2) * G + g] = a;
                ws[(2u * c2 + 1u) * G + g] = b;
            }
        } else {
            let g = li / 4u;
            let kq = (li % 4u) * 4u;
            var t0 = vec4<f32>(0.0);
            var t1 = vec4<f32>(0.0);
            var t2 = vec4<f32>(0.0);
            var t3 = vec4<f32>(0.0);
            for (var j = 0u; j < 4u; j++) {
                let gn = col0 + 4u * g + j;
                if (gn < p.n) {
                    let e = gn * p.k + k0 + kq;
                    let blk = e / 32u;
                    let d = block_scale(sc[blk / 2u], blk);
                    var v: vec4<f32>;
                    if (QUANT == 1u) {
                        v = dq8(w[e / 4u], d);
                    } else {
                        let q = w[blk * 4u + ((e % 32u) % 16u) / 4u];
                        if ((e % 32u) < 16u) { v = dq4lo(q, d); } else { v = dq4hi(q, d); }
                    }
                    t0[j] = v.x; t1[j] = v.y; t2[j] = v.z; t3[j] = v.w;
                }
            }
            ws[kq * G + g] = t0;
            ws[(kq + 1u) * G + g] = t1;
            ws[(kq + 2u) * G + g] = t2;
            ws[(kq + 3u) * G + g] = t3;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let xa = xs[kk * G + 2u * tr];
            let xb = xs[kk * G + 2u * tr + 1u];
            let wlo = ws[kk * G + 2u * tc];
            let whi = ws[kk * G + 2u * tc + 1u];
            lo0 += xa.x * wlo; hi0 += xa.x * whi;
            lo1 += xa.y * wlo; hi1 += xa.y * whi;
            lo2 += xa.z * wlo; hi2 += xa.z * whi;
            lo3 += xa.w * wlo; hi3 += xa.w * whi;
            lo4 += xb.x * wlo; hi4 += xb.x * whi;
            lo5 += xb.y * wlo; hi5 += xb.y * whi;
            lo6 += xb.z * wlo; hi6 += xb.z * whi;
            lo7 += xb.w * wlo; hi7 += xb.w * whi;
        }
        workgroupBarrier();
    }

    let r = row0 + tr * 8u;
    let c = col0 + tc * 8u;
    store_row(r, c, lo0, hi0);
    store_row(r + 1u, c, lo1, hi1);
    store_row(r + 2u, c, lo2, hi2);
    store_row(r + 3u, c, lo3, hi3);
    store_row(r + 4u, c, lo4, hi4);
    store_row(r + 5u, c, lo5, hi5);
    store_row(r + 6u, c, lo6, hi6);
    store_row(r + 7u, c, lo7, hi7);
}

// Eight consecutive columns c..c+8 of row r.
fn store_row(r: u32, c: u32, lo: vec4<f32>, hi: vec4<f32>) {
    if (r >= p.m) { return; }
    let base = r * p.n + c;
    if (c + 8u <= p.n) {
        y[base] = lo.x; y[base + 1u] = lo.y; y[base + 2u] = lo.z; y[base + 3u] = lo.w;
        y[base + 4u] = hi.x; y[base + 5u] = hi.y; y[base + 6u] = hi.z; y[base + 7u] = hi.w;
        return;
    }
    if (c < p.n) { y[base] = lo.x; }
    if (c + 1u < p.n) { y[base + 1u] = lo.y; }
    if (c + 2u < p.n) { y[base + 2u] = lo.z; }
    if (c + 3u < p.n) { y[base + 3u] = lo.w; }
    if (c + 4u < p.n) { y[base + 4u] = hi.x; }
    if (c + 5u < p.n) { y[base + 5u] = hi.y; }
    if (c + 6u < p.n) { y[base + 6u] = hi.z; }
    if (c + 7u < p.n) { y[base + 7u] = hi.w; }
}
