// x *= sigmoid(gate)

extern "C" __global__ void sigmoid_mul(float* x, const float* gate, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.len) {
        x[i] *= sigmoid(gate[i]);
    }
}
