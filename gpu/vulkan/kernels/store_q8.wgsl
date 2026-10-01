// cache[p.offset..][..p.len] = src, quantized a block of 32 at a time as
// `quantize` in `lib.rs` does: the scale is the largest magnitude over 127
// rounded to f16, and each activation the nearest multiple of it, ties away
// from zero, as an 8-bit integer, four to a word. The multiples are settled
// by comparing with the midpoints between them, which are exact, since the
// division that guesses them is not correctly rounded. Two blocks' scales
// share a word, so each thread takes the blocks of one word of scales, and
// keeps the half of it that belongs to a block outside the store.

struct Params {
    // Both in activations, and whole blocks.
    offset: u32,
    len: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> quants: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;

const BLOCK: u32 = 32u;

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

// The value of the bits of a non-negative half-precision float, by hand,
// since a driver may flush subnormal ones to zero.
fn from_f16(h: u32) -> f32 {
    let e = h >> 10u;
    let m = h & 0x3ffu;
    if e == 0u {
        return f32(m) * bitcast<f32>(103u << 23u);
    }
    return bitcast<f32>(((e + 112u) << 23u) | (m << 13u));
}

// Quantizes block `b` of the cache from `src[first..]`, and returns the
// bits of its scale.
fn quantize(b: u32, first: u32) -> u32 {
    var amax = 0.0;
    for (var i = 0u; i < BLOCK; i++) {
        amax = max(amax, abs(src[first + i]));
    }
    let bits = to_f16(amax * (1.0 / 127.0));
    let d = from_f16(bits);
    for (var w = 0u; w < BLOCK / 4u; w++) {
        var word = 0u;
        for (var j = 0u; j < 4u; j++) {
            let x = src[first + 4u * w + j];
            let a = abs(x);
            var n = 0.0;
            if d > 0.0 {
                n = min(round(a / d), 128.0);
                if a >= (n + 0.5) * d {
                    n += 1.0;
                } else if n > 0.0 && a < (n - 0.5) * d {
                    n -= 1.0;
                }
                n = min(n, 127.0);
            }
            var q = i32(n);
            if x < 0.0 {
                q = -q;
            }
            word |= (bitcast<u32>(q) & 0xffu) << (8u * j);
        }
        quants[b * (BLOCK / 4u) + w] = word;
    }
    return bits;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let first = p.offset / BLOCK;
    let end = first + p.len / BLOCK;
    let word = first / 2u + gid.x;
    if 2u * word >= end {
        return;
    }
    var scale = scales[word];
    for (var h = 0u; h < 2u; h++) {
        let b = 2u * word + h;
        if b >= first && b < end {
            let bits = quantize(b, (b - first) * BLOCK);
            scale = (scale & ~(0xffffu << (16u * h))) | (bits << (16u * h));
        }
    }
    scales[word] = scale;
}
