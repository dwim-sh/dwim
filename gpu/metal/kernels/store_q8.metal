// cache[p.offset..][..p.len] = src, quantized a block of 32 at a time as
// `quantize` in `lib.rs` does: one SIMD group per block, each lane taking
// one element. The scale is the largest magnitude over 127 rounded to f16,
// and each activation the nearest multiple of it, ties away from zero, as
// an 8-bit integer. The multiples are settled by comparing with the
// midpoints between them, which are exact, since the division that guesses
// them may not be correctly rounded.

#include <metal_stdlib>
using namespace metal;

struct Params {
    // Both in activations, and whole blocks.
    uint offset;
    uint len;
};

constant uint BLOCK = 32;

kernel void store_q8(
    device char* quants [[buffer(0)]],
    device half* scales [[buffer(1)]],
    const device float* src [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint i [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (i >= p.len / BLOCK) {
        return;
    }
    uint b = p.offset / BLOCK + i;
    float v = src[i * BLOCK + lane];
    float a = fabs(v);
    float amax = simd_max(a);
    half scale = half(min(amax * (1.0f / 127.0f), 65504.0f));
    float d = float(scale);
    float n = 0.0f;
    if (d > 0.0f) {
        n = min(rint(a / d), 128.0f);
        if (a >= (n + 0.5f) * d) {
            n += 1.0f;
        } else if (n > 0.0f && a < (n - 0.5f) * d) {
            n -= 1.0f;
        }
        n = min(n, 127.0f);
    }
    quants[b * BLOCK + lane] = char(v < 0.0f ? -n : n);
    if (lane == 0) {
        scales[b] = scale;
    }
}
