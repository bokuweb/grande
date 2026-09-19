// act[M, F] = gelu_tanh(X * Wg^T) * (X * Wu^T): the gate and up projections
// and the GeGLU in one kernel. Same tiling as matmul.wgsl but 128 invocations
// each owning 4 rows x 8 cols of both products; only the activation is
// written.

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> wg: array<u32>;
@group(0) @binding(3) var<storage, read> wu: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const LD: u32 = 65u;

var<workgroup> xs: array<f32, 1040>;
var<workgroup> gs: array<f32, 1040>;
var<workgroup> us: array<f32, 1040>;

fn gelu(x: vec4<f32>) -> vec4<f32> {
    // tanh(z) is +-1 to f32 precision beyond |z| ~ 15; naive GPU tanh
    // implementations overflow to NaN there, so clamp the argument.
    let z = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), vec4<f32>(-15.0), vec4<f32>(15.0));
    return 0.5 * x * (1.0 + tanh(z));
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wgid.y * BM;
    let col0 = wgid.x * BN;
    let tr = li / 8u;  // rows tr + 16i, i < 4
    let tc = li % 8u;  // cols tc + 8j, j < 8
    var g0lo = vec4<f32>(0.0); var g0hi = vec4<f32>(0.0); var u0lo = vec4<f32>(0.0); var u0hi = vec4<f32>(0.0);
    var g1lo = vec4<f32>(0.0); var g1hi = vec4<f32>(0.0); var u1lo = vec4<f32>(0.0); var u1hi = vec4<f32>(0.0);
    var g2lo = vec4<f32>(0.0); var g2hi = vec4<f32>(0.0); var u2lo = vec4<f32>(0.0); var u2hi = vec4<f32>(0.0);
    var g3lo = vec4<f32>(0.0); var g3hi = vec4<f32>(0.0); var u3lo = vec4<f32>(0.0); var u3hi = vec4<f32>(0.0);

    let ktiles = p.k / BK;
    for (var kt = 0u; kt < ktiles; kt++) {
        let k0 = kt * BK;
        // X tile: k = li % 16, rows li/16 + 8i.
        {
            let c = li % BK;
            for (var i = 0u; i < 8u; i++) {
                let r = li / BK + 8u * i;
                let gr = row0 + r;
                var v = 0.0;
                if (gr < p.m) { v = x[gr * p.k + k0 + c]; }
                xs[c * LD + r] = v;
            }
        }
        // Gate and up tiles: pair li % 8, cols li/8 + 16i.
        {
            let c2 = li % 8u;
            for (var i = 0u; i < 4u; i++) {
                let r = li / 8u + 16u * i;
                let gn = col0 + r;
                var a = vec2<f32>(0.0);
                var b = vec2<f32>(0.0);
                if (gn < p.n) {
                    let base = (gn * p.k + k0) / 2u + c2;
                    a = unpack2x16float(wg[base]);
                    b = unpack2x16float(wu[base]);
                }
                gs[(2u * c2) * LD + r] = a.x;
                gs[(2u * c2 + 1u) * LD + r] = a.y;
                us[(2u * c2) * LD + r] = b.x;
                us[(2u * c2 + 1u) * LD + r] = b.y;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let xb = kk * LD + tr;
            let wb = kk * LD + tc;
            let glo = vec4<f32>(gs[wb], gs[wb + 8u], gs[wb + 16u], gs[wb + 24u]);
            let ghi = vec4<f32>(gs[wb + 32u], gs[wb + 40u], gs[wb + 48u], gs[wb + 56u]);
            let ulo = vec4<f32>(us[wb], us[wb + 8u], us[wb + 16u], us[wb + 24u]);
            let uhi = vec4<f32>(us[wb + 32u], us[wb + 40u], us[wb + 48u], us[wb + 56u]);
            let x0 = xs[xb];
            let x1 = xs[xb + 16u];
            let x2 = xs[xb + 32u];
            let x3 = xs[xb + 48u];
            g0lo += x0 * glo; g0hi += x0 * ghi; u0lo += x0 * ulo; u0hi += x0 * uhi;
            g1lo += x1 * glo; g1hi += x1 * ghi; u1lo += x1 * ulo; u1hi += x1 * uhi;
            g2lo += x2 * glo; g2hi += x2 * ghi; u2lo += x2 * ulo; u2hi += x2 * uhi;
            g3lo += x3 * glo; g3hi += x3 * ghi; u3lo += x3 * ulo; u3hi += x3 * uhi;
        }
        workgroupBarrier();
    }

    let c = col0 + tc;
    store_row(row0 + tr, c, gelu(g0lo) * u0lo, gelu(g0hi) * u0hi);
    store_row(row0 + tr + 16u, c, gelu(g1lo) * u1lo, gelu(g1hi) * u1hi);
    store_row(row0 + tr + 32u, c, gelu(g2lo) * u2lo, gelu(g2hi) * u2hi);
    store_row(row0 + tr + 48u, c, gelu(g3lo) * u3lo, gelu(g3hi) * u3hi);
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
