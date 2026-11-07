// cache[p.offset..][..p.len] = src, quantized a block of 32 at a time as
// `quantize` in `gpu/mod.rs` does, a thread per block: the scale is the largest
// magnitude over 127 rounded to f16, and each activation the nearest
// multiple of it, ties away from zero, as an 8-bit integer. The multiples
// are settled by comparing with the midpoints between them, which are
// exact, since the division that guesses them is not correctly rounded.

extern "C" __global__ void store_q8(signed char* quants, u16* scales, const float* src, StoreParams p) {
    u32 first = p.offset / CACHE_BLOCK;
    u32 b = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= first + p.len / CACHE_BLOCK) {
        return;
    }
    const float* block = src + (b - first) * CACHE_BLOCK;
    float amax = 0.0f;
    for (u32 i = 0; i < CACHE_BLOCK; i++) {
        amax = fmaxf(amax, fabsf(block[i]));
    }
    // Clamped to the largest finite half rather than rounded to infinity.
    u32 bits = f32_to_f16(fminf(amax * (1.0f / 127.0f), 65504.0f));
    float d = f16_to_f32(bits);
    for (u32 i = 0; i < CACHE_BLOCK; i++) {
        float x = block[i];
        float a = fabsf(x);
        float n = 0.0f;
        if (d > 0.0f) {
            n = fminf(rintf(a / d), 128.0f);
            if (a >= (n + 0.5f) * d) {
                n += 1.0f;
            } else if (n > 0.0f && a < (n - 0.5f) * d) {
                n -= 1.0f;
            }
            n = fminf(n, 127.0f);
        }
        i32 q = (i32)n;
        quants[b * CACHE_BLOCK + i] = (signed char)(x < 0.0f ? -q : q);
    }
    scales[b] = (u16)bits;
}
