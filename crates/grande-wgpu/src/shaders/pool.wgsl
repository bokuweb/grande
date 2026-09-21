// Mean pooling over a packed batch, L2-normalised: out[slot, :] = the mean
// of x over sequence s's rows, divided by its norm. One workgroup per
// sequence, invocations striding over d; `spans` holds (first row, last
// row + 1, output slot) per sequence. The sentence embedding of an e5 /
// BERT encoder (e5.rs).

struct Params { n: u32, d: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> spans: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let s = wg.x;
    if (s >= p.n) { return; }
    let lo = spans[3u * s];
    let hi = spans[3u * s + 1u];
    let slot = spans[3u * s + 2u];
    let inv = 1.0 / f32(max(hi - lo, 1u));
    var sq = 0.0;
    for (var c = l.x; c < p.d; c += 256u) {
        var acc = 0.0;
        for (var r = lo; r < hi; r++) { acc += x[r * p.d + c]; }
        let m = acc * inv;
        out[slot * p.d + c] = m;
        sq += m * m;
    }
    red[l.x] = sq;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (l.x < st) { red[l.x] += red[l.x + st]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(max(red[0], 1e-24));
    for (var c = l.x; c < p.d; c += 256u) {
        out[slot * p.d + c] *= scale;
    }
}
