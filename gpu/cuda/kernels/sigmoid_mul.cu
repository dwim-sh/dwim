// x *= sigmoid(gate), and the result as halves too if `halves` is there
// to take them, for the matmul that follows.

extern "C" __global__ void sigmoid_mul(float* x, u16* halves, const float* gate, LenParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.len) {
        float v = x[i] * sigmoid(gate[i]);
        x[i] = v;
        if (halves) {
            halves[i] = f32_to_finite_f16(v);
        }
    }
}
