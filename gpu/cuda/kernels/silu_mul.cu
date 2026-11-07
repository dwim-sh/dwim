// gate = silu(gate) * up

extern "C" __global__ void silu_mul(float* gate, const float* up, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.len) {
        float g = gate[i];
        gate[i] = g * sigmoid(g) * up[i];
    }
}
