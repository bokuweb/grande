// Y[M, N] = X[M, K] * W[N, K]^T   (a linear layer; W row-major as HF stores it)
// X, Y are f32; W is f16 packed two per u32 or a quantized block format (see
// weight.wgsl, prepended; QUANT picks the decoder). A 64x128 output tile per
// workgroup of 128 invocations, each owning 8 consecutive rows x 8 consecutive
// cols. K streams through workgroup memory 32 wide (one quantized block per
// column per step) as f16 pairs packed over k: xs[kp][row] and ws[kp][col]
// hold (k, k+1) of one row / column, so a step of the inner loop is 4 vec4
// loads and 16 unpacks for 128 FMAs, and the tiles are half the bytes of f32
// ones. X is rounded to f16 on the way in (as llama.cpp's Metal mat-mul does).
// Measured on an M4 and rejected: an f32 X tile (+30%: the extra workgroup
// bandwidth costs more than the unpacks) and prefetching the next tile into
// registers before the multiply (+20-140%: the live registers cut occupancy).
// K must be a multiple of 32; M and N are bounds-checked.

struct Params { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> sc: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 128u;
const BK: u32 = 32u;
const KP: u32 = BK / 2u;   // k pairs per tile
const XG: u32 = BM / 4u;   // vec4<u32> per kp row of xs
const WG: u32 = BN / 4u;   // vec4<u32> per kp row of ws

var<workgroup> xs: array<vec4<u32>, KP * XG>;  // rows 4g..4g+4 at (k, k+1)
var<workgroup> ws: array<vec4<u32>, KP * WG>;  // cols 4g..4g+4 at (k, k+1)

// One invocation's share of a tile between the global loads and the
// workgroup stores. X: one k pair of two row quads. W (invocations 0..64
// only): four consecutive columns over half the k pairs, in storage form.
// Every workgroup store is a whole vec4: a component store to workgroup
// memory may compile to a read-modify-write of the vector and race with the
// other lanes' stores (it did, on Metal).
struct Stage {
    xr: array<vec4<u32>, 2>,
    wq: array<vec4<u32>, 8>,  // f16: 2 per column; Q4 / Q8: 1 per column
    d: vec4<f32>,
}

fn load_stage(li: u32, row0: u32, col0: u32, k0: u32) -> Stage {
    var s: Stage;
    // X: k pair li % 16 of row quads li / 16 + 8i; a simdgroup reads two
    // quads' 128 contiguous bytes per row.
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
    // W: columns 4g..4g+4 (g = li / 2), k pairs 8h..8h+8 (h = li % 2).
    s.d = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) { s.wq[i] = vec4<u32>(0u); }
    if (li < 64u) {
        let g = li / 2u;
        let h = li % 2u;
        for (var j = 0u; j < 4u; j++) {
            let n = col0 + 4u * g + j;
            if (n >= p.n) { continue; }
            let e = n * p.k + k0;
            if (QUANT == 0u) {
                let base = e / 8u + 2u * h;
                s.wq[2u * j] = w[base];
                s.wq[2u * j + 1u] = w[base + 1u];
            } else {
                let blk = e / 32u;
                s.d[j] = block_scale(sc[blk / 2u], blk);
                if (QUANT == 1u) {
                    s.wq[j] = w[blk * 2u + h];
                } else {
                    s.wq[j] = w[blk];
                }
            }
        }
    }
    return s;
}

// Column j of the stage at k pair jj (0..8) of its half h, as an f16 pair.
fn stage_pair(s: Stage, h: u32, j: u32, jj: u32) -> u32 {
    if (QUANT == 0u) {
        return s.wq[2u * j + jj / 4u][jj % 4u];
    }
    if (QUANT == 1u) {
        // elements 16h + 2jj, +1: word jj / 2 of the staged vec4
        return dq8_pair(s.wq[j][jj / 2u], (jj % 2u) * 16u, s.d[j]);
    }
    // Q4: half 0 is the low nibbles, half 1 the high nibbles of word jj / 2.
    return dq4_pair(s.wq[j][jj / 2u], (jj % 2u) * 16u + 4u * h, s.d[j]);
}

fn store_stage(li: u32, s: Stage) {
    let kp = li % KP;
    for (var i = 0u; i < 2u; i++) {
        xs[kp * XG + li / KP + 8u * i] = s.xr[i];
    }
    if (li < 64u) {
        let g = li / 2u;
        let h = li % 2u;
        for (var jj = 0u; jj < 8u; jj++) {
            ws[(8u * h + jj) * WG + g] = vec4<u32>(
                stage_pair(s, h, 0u, jj),
                stage_pair(s, h, 1u, jj),
                stage_pair(s, h, 2u, jj),
                stage_pair(s, h, 3u, jj),
            );
        }
    }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    let tr = li / 16u; // rows tr*8 .. tr*8+8
    let tc = li % 16u; // cols tc*8 .. tc*8+8
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
        store_stage(li, load_stage(li, row0, col0, kt * BK));
        workgroupBarrier();
        for (var kp = 0u; kp < KP; kp++) {
            let xa = xs[kp * XG + 2u * tr];
            let xb = xs[kp * XG + 2u * tr + 1u];
            let wa = ws[kp * WG + 2u * tc];
            let wb = ws[kp * WG + 2u * tc + 1u];
            let c0 = unpack2x16float(wa.x);
            let c1 = unpack2x16float(wa.y);
            let c2 = unpack2x16float(wa.z);
            let c3 = unpack2x16float(wa.w);
            let c4 = unpack2x16float(wb.x);
            let c5 = unpack2x16float(wb.y);
            let c6 = unpack2x16float(wb.z);
            let c7 = unpack2x16float(wb.w);
            // cols tc*8.. at k (lo_k / hi_k) and at k + 1 (lo_n / hi_n)
            let wlo_k = vec4<f32>(c0.x, c1.x, c2.x, c3.x);
            let wlo_n = vec4<f32>(c0.y, c1.y, c2.y, c3.y);
            let whi_k = vec4<f32>(c4.x, c5.x, c6.x, c7.x);
            let whi_n = vec4<f32>(c4.y, c5.y, c6.y, c7.y);
            let r0 = unpack2x16float(xa.x);
            let r1 = unpack2x16float(xa.y);
            let r2 = unpack2x16float(xa.z);
            let r3 = unpack2x16float(xa.w);
            let r4 = unpack2x16float(xb.x);
            let r5 = unpack2x16float(xb.y);
            let r6 = unpack2x16float(xb.z);
            let r7 = unpack2x16float(xb.w);
            lo0 += r0.x * wlo_k + r0.y * wlo_n; hi0 += r0.x * whi_k + r0.y * whi_n;
            lo1 += r1.x * wlo_k + r1.y * wlo_n; hi1 += r1.x * whi_k + r1.y * whi_n;
            lo2 += r2.x * wlo_k + r2.y * wlo_n; hi2 += r2.x * whi_k + r2.y * whi_n;
            lo3 += r3.x * wlo_k + r3.y * wlo_n; hi3 += r3.x * whi_k + r3.y * whi_n;
            lo4 += r4.x * wlo_k + r4.y * wlo_n; hi4 += r4.x * whi_k + r4.y * whi_n;
            lo5 += r5.x * wlo_k + r5.y * wlo_n; hi5 += r5.x * whi_k + r5.y * whi_n;
            lo6 += r6.x * wlo_k + r6.y * wlo_n; hi6 += r6.x * whi_k + r6.y * whi_n;
            lo7 += r7.x * wlo_k + r7.y * wlo_n; hi7 += r7.x * whi_k + r7.y * whi_n;
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
