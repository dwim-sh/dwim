// cache[p.offset..][..p.len] = src, quantized a block of 32 at a time as
// `quantize` in `lib.rs` does: one thread per block. The scale is the largest
// magnitude over 127 rounded to f16, and each activation the nearest
// multiple of it, ties away from zero, as an 8-bit integer. The multiples
// are settled by comparing with the midpoints between them, which are
// exact, since the division that guesses them may not be correctly rounded.

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
    uint i [[thread_position_in_grid]])
{
    if (i >= p.len / BLOCK) {
        return;
    }
    const device float* x = src + i * BLOCK;
    uint b = p.offset / BLOCK + i;
    float amax = 0.0f;
    for (uint j = 0; j < BLOCK; j++) {
        amax = max(amax, fabs(x[j]));
    }
    half scale = half(min(amax * (1.0f / 127.0f), 65504.0f));
    float d = float(scale);
    for (uint j = 0; j < BLOCK; j++) {
        float a = fabs(x[j]);
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
        quants[b * BLOCK + j] = char(x[j] < 0.0f ? -n : n);
    }
    scales[b] = scale;
}
