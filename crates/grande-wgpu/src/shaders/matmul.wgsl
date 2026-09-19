// Y[M, N] = X[M, K] * W[N, K]^T   (a linear layer; W row-major as HF stores it)
// X, Y are f32; W is f16 packed two per u32. 64x64 output tile per workgroup,
// 16-wide K tiles through workgroup memory, 4x4 outputs per invocation.
// Tiles are stored k-major with a padded stride so that both the transposed
// stores and the inner-loop reads are bank-conflict free. K must be a
// multiple of 16; M and N are bounds-checked.

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const LD: u32 = 65u; // padded tile stride

var<workgroup> xs: array<f32, 1040>; // [BK][LD]
var<workgroup> ws: array<f32, 1040>; // [BK][LD]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    // This invocation owns rows {tr, tr+16, tr+32, tr+48} x cols {tc, ...}.
    let tr = li / 16u;
    let tc = li % 16u;
    // Accumulators as vectors, never dynamically indexed, so they stay in
    // registers.
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ktiles = p.k / BK;
    for (var kt = 0u; kt < ktiles; kt++) {
        let k0 = kt * BK;
        // X tile: 64 rows x 16 k, four consecutive k per invocation.
        {
            let r = li / 4u;
            let c = (li % 4u) * 4u;
            let gr = row0 + r;
            var v = vec4<f32>(0.0);
            if (gr < p.m) {
                let base = gr * p.k + k0 + c;
                v = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
            }
            xs[(c + 0u) * LD + r] = v.x;
            xs[(c + 1u) * LD + r] = v.y;
            xs[(c + 2u) * LD + r] = v.z;
            xs[(c + 3u) * LD + r] = v.w;
        }
        // W tile: 64 cols x 16 k, four consecutive k (two packed pairs).
        {
            let r = li / 4u;
            let c = (li % 4u) * 4u;
            let gn = col0 + r;
            var a = vec2<f32>(0.0);
            var b = vec2<f32>(0.0);
            if (gn < p.n) {
                let base = (gn * p.k + k0 + c) / 2u;
                a = unpack2x16float(w[base]);
                b = unpack2x16float(w[base + 1u]);
            }
            ws[(c + 0u) * LD + r] = a.x;
            ws[(c + 1u) * LD + r] = a.y;
            ws[(c + 2u) * LD + r] = b.x;
            ws[(c + 3u) * LD + r] = b.y;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let xb = kk * LD + tr;
            let wb = kk * LD + tc;
            let wv = vec4<f32>(ws[wb], ws[wb + 16u], ws[wb + 32u], ws[wb + 48u]);
            acc0 += xs[xb] * wv;
            acc1 += xs[xb + 16u] * wv;
            acc2 += xs[xb + 32u] * wv;
            acc3 += xs[xb + 48u] * wv;
        }
        workgroupBarrier();
    }

    store_row(row0 + tr, col0 + tc, acc0);
    store_row(row0 + tr + 16u, col0 + tc, acc1);
    store_row(row0 + tr + 32u, col0 + tc, acc2);
    store_row(row0 + tr + 48u, col0 + tc, acc3);
}

fn store_row(r: u32, c: u32, v: vec4<f32>) {
    if (r >= p.m) { return; }
    let base = r * p.n + c;
    if (c < p.n) { y[base] = v.x; }
    if (c + 16u < p.n) { y[base + 16u] = v.y; }
    if (c + 32u < p.n) { y[base + 32u] = v.z; }
    if (c + 48u < p.n) { y[base + 48u] = v.w; }
}
