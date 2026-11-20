// out[t][r] = w[r] · x[t]: one block of 64 threads per row of the bf16
// weight matrix and group of eight tokens, reading the row once as 16-byte
// chunks, eight weights at a time, and taking each chunk's dot product
// with the eight tokens' activations, so that a batch of tokens costs the
// row's reads once and no more barriers than one token does.

// Tokens a block takes.
#define BF16_TOKENS 8

__device__ __forceinline__ float bf16_lo(u32 pair) {
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float bf16_hi(u32 pair) {
    return __uint_as_float(pair & 0xffff0000u);
}

extern "C" __global__ void __launch_bounds__(64) matmul_bf16(float* out, const uint4* w, const float4* x, MatmulParams p) {
    __shared__ float partial[2][BF16_TOKENS];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    u32 token_groups = (p.n + BF16_TOKENS - 1) / BF16_TOKENS;
    u32 r = (blockIdx.y / token_groups) * p.stride + blockIdx.x;
    u32 t0 = (blockIdx.y % token_groups) * BF16_TOKENS;
    u32 count = min((u32)BF16_TOKENS, p.n - t0);
    u32 chunks = p.cols / 8;
    u32 base = r * chunks;
    float acc[BF16_TOKENS];
#pragma unroll
    for (u32 t = 0; t < BF16_TOKENS; t++) {
        acc[t] = 0.0f;
    }
    if (r < p.rows) {
        for (u32 c = lid; c < chunks; c += 64) {
            uint4 wv = w[base + c];
            float ws[8] = {
                bf16_lo(wv.x), bf16_hi(wv.x), bf16_lo(wv.y), bf16_hi(wv.y),
                bf16_lo(wv.z), bf16_hi(wv.z), bf16_lo(wv.w), bf16_hi(wv.w),
            };
#pragma unroll
            for (u32 t = 0; t < BF16_TOKENS; t++) {
                if (t < count) {
                    u32 xbase = (t0 + t) * (p.cols / 4) + c * 2;
                    float4 x0 = x[xbase];
                    float4 x1 = x[xbase + 1];
                    acc[t] += ws[0] * x0.x + ws[1] * x0.y + ws[2] * x0.z + ws[3] * x0.w
                        + ws[4] * x1.x + ws[5] * x1.y + ws[6] * x1.z + ws[7] * x1.w;
                }
            }
        }
    }
#pragma unroll
    for (u32 t = 0; t < BF16_TOKENS; t++) {
        float s = warp_sum(acc[t]);
        if (lane == 0) {
            partial[warp][t] = s;
        }
    }
    __syncthreads();
    if (lid < count && r < p.rows) {
        out[(t0 + lid) * p.rows + r] = partial[0][lid] + partial[1][lid];
    }
}
