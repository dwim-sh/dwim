// cache[p.offset..][..p.len] = f16(src): each thread packs one pair of
// activations into a word of the cache as two half-precision floats, rounded
// to the nearest, ties to even, and clamped to the largest finite one. The
// rounding is done by hand because `pack2x16float` rounds toward zero on
// AMD's driver, which is allowed, and the CPU rounds to the nearest.

struct Params {
    // Both in activations, and even.
    offset: u32,
    len: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> cache: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;

// The bits of the half-precision float nearest to `x`, ties to even, with
// `x` clamped to the largest finite one.
fn to_f16(x: f32) -> u32 {
    let sign = (bitcast<u32>(x) >> 16u) & 0x8000u;
    let bits = bitcast<u32>(min(abs(x), 65504.0));
    // Below 2^-14 the half is subnormal, a multiple of 2^-24, and rounding
    // may carry it up to the smallest normal, whose bits follow on from the
    // largest subnormal's.
    if bits < (113u << 23u) {
        return sign | u32(round(bitcast<f32>(bits) * 16777216.0));
    }
    // Rebase the exponent from f32's 127 to f16's 15, then round the 13
    // mantissa bits away to the nearest, ties to even; a carry runs into
    // the exponent.
    let rebased = bits - (112u << 23u);
    let rest = rebased & 0x1fffu;
    var half = rebased >> 13u;
    if rest > 0x1000u || (rest == 0x1000u && (half & 1u) == 1u) {
        half += 1u;
    }
    return sign | half;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.len / 2u {
        return;
    }
    cache[p.offset / 2u + i] = to_f16(src[2u * i]) | (to_f16(src[2u * i + 1u]) << 16u);
}
