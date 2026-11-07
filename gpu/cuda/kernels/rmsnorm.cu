// Normalizes each row of x by its root mean square and scales it by the
// weight: one block of 256 threads per row. The row is read as vectors of
// four, a fixed number per thread with the index clamped rather than the
// load skipped, so that the loads go out together instead of one round
// trip after another.

// Vectors a thread reads: rows of up to 8192 elements.
#define RMSNORM_PER_THREAD 8

extern "C" __global__ void __launch_bounds__(256) rmsnorm(float4* x, const float4* weight, RmsnormParams p) {
    __shared__ float partial[8];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    u32 vectors = p.dim / 4;
    u32 base = blockIdx.x * vectors;
    float4 vs[RMSNORM_PER_THREAD];
    float ss = 0.0f;
#pragma unroll
    for (u32 k = 0; k < RMSNORM_PER_THREAD; k++) {
        u32 i = lid + k * 256;
        vs[k] = x[base + min(i, vectors - 1)];
        ss += i < vectors ? dot4(vs[k], vs[k]) : 0.0f;
    }
    float s = warp_sum(ss);
    if (lane == 0) {
        partial[warp] = s;
    }
    __syncthreads();
    float total = 0.0f;
    for (u32 i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrtf(total / (float)p.dim + p.eps);
#pragma unroll
    for (u32 k = 0; k < RMSNORM_PER_THREAD; k++) {
        u32 i = lid + k * 256;
        if (i < vectors) {
            x[base + i] = mul4(scale4(vs[k], scale), weight[i]);
        }
    }
}
