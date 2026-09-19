// Weight decoding shared by every kernel that reads a weight matrix. Prepended
// to the kernel source by the engine. QUANT selects the storage type of the
// bound weight (model::Dtype::code): 0 = f16 pairs, 1 = Q8_0, 2 = Q4_0. The
// quantized types are 32-element blocks with one f16 scale per block; the
// scales live in a separate buffer (`sc`, f16 pairs) so the payload stays
// 4-byte aligned. Q8: 32 signed bytes. Q4: 16 bytes, low nibbles are elements
// 0..16, high nibbles 16..32, value (q - 8) * scale.

override QUANT: u32 = 0u;

// Scale of block `blk` from a scales buffer word.
fn block_scale(word: u32, blk: u32) -> f32 {
    let v = unpack2x16float(word);
    return select(v.x, v.y, (blk & 1u) == 1u);
}

// Four consecutive int8 codes of one word, times the block scale.
fn dq8(q: u32, d: f32) -> vec4<f32> {
    let v = vec4<i32>(
        i32(q << 24u) >> 24u,
        i32(q << 16u) >> 24u,
        i32(q << 8u) >> 24u,
        i32(q) >> 24u,
    );
    return vec4<f32>(v) * d;
}

// Low nibbles (elements 4u..4u+4 of the block's first half) of one word.
fn dq4lo(q: u32, d: f32) -> vec4<f32> {
    let v = vec4<u32>(q & 0xfu, (q >> 8u) & 0xfu, (q >> 16u) & 0xfu, (q >> 24u) & 0xfu);
    return (vec4<f32>(v) - 8.0) * d;
}

// High nibbles (elements 16+4u.. of the block) of one word.
fn dq4hi(q: u32, d: f32) -> vec4<f32> {
    let v = vec4<u32>((q >> 4u) & 0xfu, (q >> 12u) & 0xfu, (q >> 20u) & 0xfu, (q >> 28u) & 0xfu);
    return (vec4<f32>(v) - 8.0) * d;
}
