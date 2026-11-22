// Normalizes each row of x by its root mean square and scales it by the
// weight, in place, or into `out` as halves: one block of 256 threads per
// row. The row is read as vectors of
// four, a fixed number per thread with the index clamped rather than the
// load skipped, so that the loads go out together instead of one round
// trip after another.

// Vectors a thread reads: rows of up to 8192 elements.
#define RMSNORM_PER_THREAD 8

template <bool HALVES>
__device__ __forceinline__ void rmsnorm_body(void* out, const float4* x, const float4* weight, RmsnormParams p) {
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
            float4 v = mul4(scale4(vs[k], scale), weight[i]);
            if (HALVES) {
                ((uint2*)out)[base + i] = halves4(v);
            } else {
                ((float4*)out)[base + i] = v;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) rmsnorm(float4* x, const float4* weight, RmsnormParams p) {
    rmsnorm_body<false>(x, x, weight, p);
}

extern "C" __global__ void __launch_bounds__(256) normalize_f16(
    uint2* out, const float4* x, const float4* weight, RmsnormParams p) {
    rmsnorm_body<true>(out, x, weight, p);
}

// Several rows of `normalize_f16` in one grid, each job's own input,
// weight and output: the rows across the grid and the jobs down it.
#define NORM_JOBS 4

struct NormJobs {
    uint2* out[NORM_JOBS];
    const float4* x[NORM_JOBS];
    const float4* weight[NORM_JOBS];
    u32 dim;
    float eps;
};

extern "C" __global__ void __launch_bounds__(256) normalizes_f16(NormJobs p) {
    RmsnormParams one = {p.dim, p.eps};
    rmsnorm_body<true>(p.out[blockIdx.y], p.x[blockIdx.y], p.weight[blockIdx.y], one);
}
