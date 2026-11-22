// Normalizes each row of x by its root mean square and scales it by the
// weight, in place, or into `out` as halves: one block of 256 threads per
// row. The row is read as vectors of four, a thread looping over its share
// of them, so that a row of any length is handled; a length not a multiple
// of four leaves a tail of scalars that the first threads of the block take.

template <bool HALVES>
__device__ __forceinline__ void rmsnorm_body(void* out, const float4* x, const float4* weight, RmsnormParams p) {
    __shared__ float partial[8];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    const u32 vectors = p.dim / 4;
    const u32 tail = p.dim - vectors * 4;
    const u32 base = blockIdx.x * vectors;
    float ss = 0.0f;
    for (u32 i = lid; i < vectors; i += 256) {
        float4 v = x[base + i];
        ss += dot4(v, v);
    }
    if (tail && lid < tail) {
        float v = ((const float*)x)[base * 4 + vectors * 4 + lid];
        ss += v * v;
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
    for (u32 i = lid; i < vectors; i += 256) {
        float4 v = mul4(scale4(x[base + i], scale), weight[i]);
        if (HALVES) {
            ((uint2*)out)[base + i] = halves4(v);
        } else {
            ((float4*)out)[base + i] = v;
        }
    }
    if (tail && lid < tail) {
        float v = ((const float*)x)[base * 4 + vectors * 4 + lid];
        float w = ((const float*)weight)[vectors * 4 + lid];
        float r = v * scale * w;
        if (HALVES) {
            ((u16*)out)[base * 4 + vectors * 4 + lid] = (u16)f32_to_finite_f16(r);
        } else {
            ((float*)out)[base * 4 + vectors * 4 + lid] = r;
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
