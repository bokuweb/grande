// act[M, F] = gelu_tanh(X * Wg^T) * (X * Wu^T): the gate and up projections
// and the GeGLU in one kernel. Same k-major vec4 tiling as matmul.wgsl but
// 128 invocations each owning 4 consecutive rows x 8 consecutive cols of both
// products; only the activation is written. Both weights share one storage
// type (QUANT, see weight.wgsl).

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> wg: array<u32>;
@group(0) @binding(3) var<storage, read> sg: array<u32>;
@group(0) @binding(4) var<storage, read> wu: array<u32>;
@group(0) @binding(5) var<storage, read> su: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const G: u32 = 16u;

var<workgroup> xs: array<vec4<f32>, 256>; // [BK][G]
var<workgroup> gs: array<vec4<f32>, 256>;
var<workgroup> us: array<vec4<f32>, 256>;

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
    let tr = li / 8u; // rows tr*4 .. tr*4+4
    let tc = li % 8u; // cols tc*8 .. tc*8+8
    var g0lo = vec4<f32>(0.0); var g0hi = vec4<f32>(0.0); var u0lo = vec4<f32>(0.0); var u0hi = vec4<f32>(0.0);
    var g1lo = vec4<f32>(0.0); var g1hi = vec4<f32>(0.0); var u1lo = vec4<f32>(0.0); var u1hi = vec4<f32>(0.0);
    var g2lo = vec4<f32>(0.0); var g2hi = vec4<f32>(0.0); var u2lo = vec4<f32>(0.0); var u2hi = vec4<f32>(0.0);
    var g3lo = vec4<f32>(0.0); var g3hi = vec4<f32>(0.0); var u3lo = vec4<f32>(0.0); var u3hi = vec4<f32>(0.0);

    let ktiles = p.k / BK;
    for (var kt = 0u; kt < ktiles; kt++) {
        let k0 = kt * BK;
        // X tile: k = li % 16, row groups li/16 + 8i.
        {
            let c = li % BK;
            for (var i = 0u; i < 2u; i++) {
                let g = li / BK + 8u * i;
                var v = vec4<f32>(0.0);
                for (var j = 0u; j < 4u; j++) {
                    let gr = row0 + 4u * g + j;
                    if (gr < p.m) { v[j] = x[gr * p.k + k0 + c]; }
                }
                xs[c * G + g] = v;
            }
        }
        // Gate (li < 64) and up (li >= 64) tiles, one operand per half.
        let up = li >= 64u;
        let lj = li % 64u;
        if (QUANT == 0u) {
            let c2 = lj % 8u;
            for (var i = 0u; i < 2u; i++) {
                let g = lj / 8u + 8u * i;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                for (var j = 0u; j < 4u; j++) {
                    let gn = col0 + 4u * g + j;
                    if (gn < p.n) {
                        let idx = (gn * p.k + k0) / 2u + c2;
                        var v: vec2<f32>;
                        if (up) { v = unpack2x16float(wu[idx]); } else { v = unpack2x16float(wg[idx]); }
                        a[j] = v.x;
                        b[j] = v.y;
                    }
                }
                if (up) {
                    us[(2u * c2) * G + g] = a;
                    us[(2u * c2 + 1u) * G + g] = b;
                } else {
                    gs[(2u * c2) * G + g] = a;
                    gs[(2u * c2 + 1u) * G + g] = b;
                }
            }
        } else {
            let g = lj / 4u;
            let kq = (lj % 4u) * 4u;
            var t0 = vec4<f32>(0.0);
            var t1 = vec4<f32>(0.0);
            var t2 = vec4<f32>(0.0);
            var t3 = vec4<f32>(0.0);
            for (var j = 0u; j < 4u; j++) {
                let gn = col0 + 4u * g + j;
                if (gn < p.n) {
                    let e = gn * p.k + k0 + kq;
                    let blk = e / 32u;
                    var d: f32;
                    var q: u32;
                    let widx = select(e / 4u, blk * 4u + ((e % 32u) % 16u) / 4u, QUANT == 2u);
                    if (up) {
                        d = block_scale(su[blk / 2u], blk);
                        q = wu[widx];
                    } else {
                        d = block_scale(sg[blk / 2u], blk);
                        q = wg[widx];
                    }
                    var v: vec4<f32>;
                    if (QUANT == 1u) {
                        v = dq8(q, d);
                    } else if ((e % 32u) < 16u) {
                        v = dq4lo(q, d);
                    } else {
                        v = dq4hi(q, d);
                    }
                    t0[j] = v.x; t1[j] = v.y; t2[j] = v.z; t3[j] = v.w;
                }
            }
            if (up) {
                us[kq * G + g] = t0;
                us[(kq + 1u) * G + g] = t1;
                us[(kq + 2u) * G + g] = t2;
                us[(kq + 3u) * G + g] = t3;
            } else {
                gs[kq * G + g] = t0;
                gs[(kq + 1u) * G + g] = t1;
                gs[(kq + 2u) * G + g] = t2;
                gs[(kq + 3u) * G + g] = t3;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let xa = xs[kk * G + tr];
            let glo = gs[kk * G + 2u * tc];
            let ghi = gs[kk * G + 2u * tc + 1u];
            let ulo = us[kk * G + 2u * tc];
            let uhi = us[kk * G + 2u * tc + 1u];
            g0lo += xa.x * glo; g0hi += xa.x * ghi; u0lo += xa.x * ulo; u0hi += xa.x * uhi;
            g1lo += xa.y * glo; g1hi += xa.y * ghi; u1lo += xa.y * ulo; u1hi += xa.y * uhi;
            g2lo += xa.z * glo; g2hi += xa.z * ghi; u2lo += xa.z * ulo; u2hi += xa.z * uhi;
            g3lo += xa.w * glo; g3hi += xa.w * ghi; u3lo += xa.w * ulo; u3hi += xa.w * uhi;
        }
        workgroupBarrier();
    }

    let r = row0 + tr * 4u;
    let c = col0 + tc * 8u;
    store_row(r, c, gelu(g0lo) * u0lo, gelu(g0hi) * u0hi);
    store_row(r + 1u, c, gelu(g1lo) * u1lo, gelu(g1hi) * u1hi);
    store_row(r + 2u, c, gelu(g2lo) * u2lo, gelu(g2hi) * u2hi);
    store_row(r + 3u, c, gelu(g3lo) * u3lo, gelu(g3hi) * u3hi);
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
