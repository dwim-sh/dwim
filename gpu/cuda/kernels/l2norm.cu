// Normalizes each row of x to unit length: one block of 256 threads per
// row.

extern "C" __global__ void __launch_bounds__(256) l2norm(float* x, RmsnormParams p) {
    __shared__ float partial[8];
    const u32 lid = threadIdx.x;
    const u32 lane = lid & 31;
    const u32 warp = lid >> 5;
    u32 base = blockIdx.x * p.dim;
    float ss = 0.0f;
    for (u32 i = lid; i < p.dim; i += 256) {
        float v = x[base + i];
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
    float scale = rsqrtf(total + p.eps);
    for (u32 i = lid; i < p.dim; i += 256) {
        x[base + i] *= scale;
    }
}
