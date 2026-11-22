// Normalizes each row of x by its root mean square and scales it by the
// weight, in place, or into `out` as halves: one block of 256 threads per
// row. A row whose length is a multiple of four is read as vectors of
// four, a thread looping over its share of them; a row whose start is not
// on a vector, which is the rest when the length is not, is read
// element by element instead.

template <bool HALVES>
__device__ __forceinline__ void rmsnorm_body(void* out, const float4* x, const float4* weight, RmsnormParams p) {
    __shared__ float partial[8];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    float ss = 0.0f;
    if (p.dim % 4 == 0) {
        const u32 vectors = p.dim / 4;
        const u32 base = blockIdx.x * vectors;
        for (u32 i = lid; i < vectors; i += 256) {
            float4 v = x[base + i];
            ss += dot4(v, v);
        }
    } else {
        const float* xf = (const float*)x;
        const u32 base = blockIdx.x * p.dim;
        for (u32 i = lid; i < p.dim; i += 256) {
            float v = xf[base + i];
            ss += v * v;
        }
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
    if (p.dim % 4 == 0) {
        const u32 vectors = p.dim / 4;
        const u32 base = blockIdx.x * vectors;
        for (u32 i = lid; i < vectors; i += 256) {
            float4 v = mul4(scale4(x[base + i], scale), weight[i]);
            if (HALVES) {
                ((uint2*)out)[base + i] = halves4(v);
            } else {
                ((float4*)out)[base + i] = v;
            }
        }
    } else {
        const float* xf = (const float*)x;
        const float* wf = (const float*)weight;
        const u32 base = blockIdx.x * p.dim;
        for (u32 i = lid; i < p.dim; i += 256) {
            float r = xf[base + i] * scale * wf[i];
            if (HALVES) {
                ((u16*)out)[base + i] = (u16)f32_to_finite_f16(r);
            } else {
                ((float*)out)[base + i] = r;
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
