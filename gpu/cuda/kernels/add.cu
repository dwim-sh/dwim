// x += y

extern "C" __global__ void add(float* x, const float* y, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.len) {
        x[i] += y[i];
    }
}
