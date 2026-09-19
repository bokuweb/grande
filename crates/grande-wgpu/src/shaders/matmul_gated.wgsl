// act[M, F] = gelu_tanh(X * Wg^T) * (X * Wu^T): the gate and up projections
// and the GeGLU in one kernel. Same f16-pair tiling as matmul.wgsl on a 64x64
// output tile: 128 invocations each own 4 consecutive rows x 8 consecutive
// cols of both products (64 accumulators). Only the activation is written.
// Both weights share one storage type (QUANT, see weight.wgsl).

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> wg: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> sg: array<u32>;
@group(0) @binding(4) var<storage, read> wu: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> su: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const KP: u32 = BK / 2u;
const G: u32 = 16u;  // vec4<u32> per kp row of every tile (BM / 4 = BN / 4)

var<workgroup> xs: array<vec4<u32>, KP * G>;
var<workgroup> gs: array<vec4<u32>, KP * G>;
var<workgroup> us: array<vec4<u32>, KP * G>;

// As matmul.wgsl: X is one k pair of two row quads per invocation; the
// weight tiles are staged four columns x half the k pairs per invocation,
// gate by invocations 0..32 and up by 32..64, with whole-vec4 stores.
struct Stage {
    xr: array<vec4<u32>, 2>,
    wq: array<vec4<u32>, 8>,
    d: vec4<f32>,
}

fn load_stage(li: u32, row0: u32, col0: u32, k0: u32) -> Stage {
    var s: Stage;
    let kp = li % KP;
    for (var i = 0u; i < 2u; i++) {
        let g = li / KP + 8u * i;
        var v = vec4<u32>(0u);
        for (var j = 0u; j < 4u; j++) {
            let r = row0 + 4u * g + j;
            if (r < p.m) { v[j] = pack2x16float(x[(r * p.k + k0) / 2u + kp]); }
        }
        s.xr[i] = v;
    }
    s.d = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) { s.wq[i] = vec4<u32>(0u); }
    if (li < 64u) {
        let up = li >= 32u;
        let g = (li % 32u) / 2u;
        let h = li % 2u;
        for (var j = 0u; j < 4u; j++) {
            let n = col0 + 4u * g + j;
            if (n >= p.n) { continue; }
            let e = n * p.k + k0;
            if (QUANT == 0u) {
                let base = e / 8u + 2u * h;
                if (up) {
                    s.wq[2u * j] = wu[base];
                    s.wq[2u * j + 1u] = wu[base + 1u];
                } else {
                    s.wq[2u * j] = wg[base];
                    s.wq[2u * j + 1u] = wg[base + 1u];
                }
            } else {
                let blk = e / 32u;
                let wi = select(blk, blk * 2u + h, QUANT == 1u);
                if (up) {
                    s.d[j] = block_scale(su[blk / 2u], blk);
                    s.wq[j] = wu[wi];
                } else {
                    s.d[j] = block_scale(sg[blk / 2u], blk);
                    s.wq[j] = wg[wi];
                }
            }
        }
    }
    return s;
}

fn stage_pair(s: Stage, h: u32, j: u32, jj: u32) -> u32 {
    if (QUANT == 0u) {
        return s.wq[2u * j + jj / 4u][jj % 4u];
    }
    if (QUANT == 1u) {
        return dq8_pair(s.wq[j][jj / 2u], (jj % 2u) * 16u, s.d[j]);
    }
    return dq4_pair(s.wq[j][jj / 2u], (jj % 2u) * 16u + 4u * h, s.d[j]);
}

fn store_stage(li: u32, s: Stage) {
    let kp = li % KP;
    for (var i = 0u; i < 2u; i++) {
        xs[kp * G + li / KP + 8u * i] = s.xr[i];
    }
    if (li < 64u) {
        let g = (li % 32u) / 2u;
        let h = li % 2u;
        for (var jj = 0u; jj < 8u; jj++) {
            let v = vec4<u32>(
                stage_pair(s, h, 0u, jj),
                stage_pair(s, h, 1u, jj),
                stage_pair(s, h, 2u, jj),
                stage_pair(s, h, 3u, jj),
            );
            if (li >= 32u) { us[(8u * h + jj) * G + g] = v; } else { gs[(8u * h + jj) * G + g] = v; }
        }
    }
}

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
        store_stage(li, load_stage(li, row0, col0, kt * BK));
        workgroupBarrier();
        for (var kp = 0u; kp < KP; kp++) {
            let xa = xs[kp * G + tr];
            let ga = gs[kp * G + 2u * tc];
            let gb = gs[kp * G + 2u * tc + 1u];
            let ua = us[kp * G + 2u * tc];
            let ub = us[kp * G + 2u * tc + 1u];
            let g0 = unpack2x16float(ga.x);
            let g1 = unpack2x16float(ga.y);
            let g2 = unpack2x16float(ga.z);
            let g3 = unpack2x16float(ga.w);
            let g4 = unpack2x16float(gb.x);
            let g5 = unpack2x16float(gb.y);
            let g6 = unpack2x16float(gb.z);
            let g7 = unpack2x16float(gb.w);
            let glo_k = vec4<f32>(g0.x, g1.x, g2.x, g3.x);
            let glo_n = vec4<f32>(g0.y, g1.y, g2.y, g3.y);
            let ghi_k = vec4<f32>(g4.x, g5.x, g6.x, g7.x);
            let ghi_n = vec4<f32>(g4.y, g5.y, g6.y, g7.y);
            let u0 = unpack2x16float(ua.x);
            let u1 = unpack2x16float(ua.y);
            let u2 = unpack2x16float(ua.z);
            let u3 = unpack2x16float(ua.w);
            let u4 = unpack2x16float(ub.x);
            let u5 = unpack2x16float(ub.y);
            let u6 = unpack2x16float(ub.z);
            let u7 = unpack2x16float(ub.w);
            let ulo_k = vec4<f32>(u0.x, u1.x, u2.x, u3.x);
            let ulo_n = vec4<f32>(u0.y, u1.y, u2.y, u3.y);
            let uhi_k = vec4<f32>(u4.x, u5.x, u6.x, u7.x);
            let uhi_n = vec4<f32>(u4.y, u5.y, u6.y, u7.y);
            let r0 = unpack2x16float(xa.x);
            let r1 = unpack2x16float(xa.y);
            let r2 = unpack2x16float(xa.z);
            let r3 = unpack2x16float(xa.w);
            g0lo += r0.x * glo_k + r0.y * glo_n; g0hi += r0.x * ghi_k + r0.y * ghi_n;
            u0lo += r0.x * ulo_k + r0.y * ulo_n; u0hi += r0.x * uhi_k + r0.y * uhi_n;
            g1lo += r1.x * glo_k + r1.y * glo_n; g1hi += r1.x * ghi_k + r1.y * ghi_n;
            u1lo += r1.x * ulo_k + r1.y * ulo_n; u1hi += r1.x * uhi_k + r1.y * uhi_n;
            g2lo += r2.x * glo_k + r2.y * glo_n; g2hi += r2.x * ghi_k + r2.y * ghi_n;
            u2lo += r2.x * ulo_k + r2.y * ulo_n; u2hi += r2.x * uhi_k + r2.y * uhi_n;
            g3lo += r3.x * glo_k + r3.y * glo_n; g3hi += r3.x * ghi_k + r3.y * ghi_n;
            u3lo += r3.x * ulo_k + r3.y * ulo_n; u3hi += r3.x * uhi_k + r3.y * uhi_n;
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
