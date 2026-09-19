// Y[M, N] = X[M, K] * W[N, K]^T   (a linear layer; W row-major as HF stores it)
// X, Y are f32; W is f16 packed two per u32. A 64x64 output tile per
// workgroup of 64 invocations, each owning 8 rows x 8 cols (rows tr+8i, cols
// tc+8j, so tile reads are consecutive across invocations). K streams through
// workgroup memory 16 wide, k-major with a padded stride: the transposed
// stores and the inner-loop reads are both bank-conflict free. K must be a
// multiple of 16; M and N are bounds-checked.

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const LD: u32 = 65u;

var<workgroup> xs: array<f32, 1040>; // [BK][LD]
var<workgroup> ws: array<f32, 1040>; // [BK][LD]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    let tr = li / 8u;
    let tc = li % 8u;
    // acc[i] = row tr+8i, cols tc+8j for j in 0..4 (lo) and 4..8 (hi)
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
        // X tile: k = li % 16, rows li/16 + 4i. Across invocations the
        // reads are 16 consecutive floats of one row (coalesced) and the
        // k-major stores hit distinct banks.
        {
            let c = li % BK;
            for (var i = 0u; i < 16u; i++) {
                let r = li / BK + 4u * i;
                let gr = row0 + r;
                var v = 0.0;
                if (gr < p.m) { v = x[gr * p.k + k0 + c]; }
                xs[c * LD + r] = v;
            }
        }
        // W tile: pair li % 8, cols li/8 + 8i.
        {
            let c2 = li % 8u;
            for (var i = 0u; i < 8u; i++) {
                let r = li / 8u + 8u * i;
                let gn = col0 + r;
                var v = vec2<f32>(0.0);
                if (gn < p.n) { v = unpack2x16float(w[(gn * p.k + k0) / 2u + c2]); }
                ws[(2u * c2) * LD + r] = v.x;
                ws[(2u * c2 + 1u) * LD + r] = v.y;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let xb = kk * LD + tr;
            let wb = kk * LD + tc;
            let wlo = vec4<f32>(ws[wb], ws[wb + 8u], ws[wb + 16u], ws[wb + 24u]);
            let whi = vec4<f32>(ws[wb + 32u], ws[wb + 40u], ws[wb + 48u], ws[wb + 56u]);
            let x0 = xs[xb];
            let x1 = xs[xb + 8u];
            let x2 = xs[xb + 16u];
            let x3 = xs[xb + 24u];
            let x4 = xs[xb + 32u];
            let x5 = xs[xb + 40u];
            let x6 = xs[xb + 48u];
            let x7 = xs[xb + 56u];
            lo0 += x0 * wlo; hi0 += x0 * whi;
            lo1 += x1 * wlo; hi1 += x1 * whi;
            lo2 += x2 * wlo; hi2 += x2 * whi;
            lo3 += x3 * wlo; hi3 += x3 * whi;
            lo4 += x4 * wlo; hi4 += x4 * whi;
            lo5 += x5 * wlo; hi5 += x5 * whi;
            lo6 += x6 * wlo; hi6 += x6 * whi;
            lo7 += x7 * wlo; hi7 += x7 * whi;
        }
        workgroupBarrier();
    }

    let c = col0 + tc;
    store_row(row0 + tr, c, lo0, hi0);
    store_row(row0 + tr + 8u, c, lo1, hi1);
    store_row(row0 + tr + 16u, c, lo2, hi2);
    store_row(row0 + tr + 24u, c, lo3, hi3);
    store_row(row0 + tr + 32u, c, lo4, hi4);
    store_row(row0 + tr + 40u, c, lo5, hi5);
    store_row(row0 + tr + 48u, c, lo6, hi6);
    store_row(row0 + tr + 56u, c, lo7, hi7);
}

fn store_row(r: u32, c: u32, lo: vec4<f32>, hi: vec4<f32>) {
    if (r >= p.m) { return; }
    let base = r * p.n + c;
    if (c < p.n) { y[base] = lo.x; }
    if (c + 8u < p.n) { y[base + 8u] = lo.y; }
    if (c + 16u < p.n) { y[base + 16u] = lo.z; }
    if (c + 24u < p.n) { y[base + 24u] = lo.w; }
    if (c + 32u < p.n) { y[base + 32u] = hi.x; }
    if (c + 40u < p.n) { y[base + 40u] = hi.y; }
    if (c + 48u < p.n) { y[base + 48u] = hi.z; }
    if (c + 56u < p.n) { y[base + 56u] = hi.w; }
}
