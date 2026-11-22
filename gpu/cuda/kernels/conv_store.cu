// A short convolution as `short_conv`'s, then each head of its output,
// `dim` long, clipped or normalized, and the head stored in a cache as
// `store_q8` stores it, token t of the batch in row `(pos + t) % rows`:
// what a token's keys and values go through, without a pass over them
// for each step. A block per head of a token, a thread a channel: the
// head's squares are added up across the block, and a warp is a block of
// the cache. Clipped, a head is scaled by `by`, a row's worth, and then
// down to a length of `value` if it is longer; normalized, it is divided
// by its root mean square, with `value` for the epsilon, and scaled by
// `by`, a head's worth.

#define KEPT_CLIPPED 0
#define KEPT_NORMED 1

extern "C" __global__ void __launch_bounds__(1024) conv_store(
    signed char* quants, u16* scales, const float* x, float* state, const float* weight,
    const float* dynamic, const float* by, ConvStoreParams p) {
    __shared__ float partial[32];
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    ShortConvParams conv = {p.n, p.channels, p.taps, p.heads, 0};
    u32 per_token = p.channels / p.dim;
    u32 t = blockIdx.x / per_token;
    u32 c = (blockIdx.x % per_token) * p.dim + threadIdx.x;
    float v = short_conv_channel(x, state, weight, dynamic, conv, t, c);
    if (p.kept == KEPT_CLIPPED) {
        v *= by[c % p.width];
    }
    float square = warp_sum(v * v);
    if (lane == 0) {
        partial[warp] = square;
    }
    __syncthreads();
    float total = 0.0f;
    for (u32 i = 0; i < p.dim / 32; i++) {
        total += partial[i];
    }
    if (p.kept == KEPT_CLIPPED) {
        v *= fminf(p.value / (sqrtf(total) + 1e-6f), 1.0f);
    } else {
        v = v * rsqrtf(total / (float)p.dim + p.value) * by[c % p.dim];
    }
    // The warp's block of the cache.
    u32 bits = f32_to_f16(fminf(warp_max(fabsf(v)) * (1.0f / 127.0f), 65504.0f));
    size_t at = (size_t)((p.pos + t) % p.rows) * p.channels + c;
    quants[at] = quantized(v, f16_to_f32(bits));
    if (lane == 0) {
        scales[at / CACHE_BLOCK] = (u16)bits;
    }
}
