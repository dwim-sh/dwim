// gate = silu(gate) * up, and the result as halves too if `halves` is
// there to take them, for the matmul that follows.

extern "C" __global__ void silu_mul(float* gate, u16* halves, const float* up, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.len) {
        float g = gate[i];
        float v = g * sigmoid(g) * up[i];
        gate[i] = v;
        if (halves) {
            halves[i] = f32_to_finite_f16(v);
        }
    }
}
