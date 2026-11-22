// The SiTU activation of the experts' feed-forward networks: each row of
// `gate_up` is a gate then an up projection, each `width` wide, and a row
// of `out` gets
// beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta),
// and the same as halves too if `halves` is there to take them, for the
// matmul that follows. And the experts' sum: out[t] is the sum over
// the token's `k` experts of each one's row of `y` times what it counts
// for. A thread an element of the output, both.

extern "C" __global__ void situ(float* out, u16* halves, const float* gate_up, SituParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.len) {
        return;
    }
    u32 row = i / p.width;
    u32 at = i % p.width;
    float g = gate_up[(size_t)row * 2 * p.width + at];
    float u = gate_up[(size_t)row * 2 * p.width + p.width + at];
    float v = situ(g, u, p.beta, p.linear_beta);
    out[i] = v;
    if (halves) {
        halves[i] = f32_to_finite_f16(v);
    }
}

extern "C" __global__ void weighted_sum(float* out, const float* y, const float* weights, RowsParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.len) {
        return;
    }
    u32 t = i / p.width;
    u32 at = i % p.width;
    float sum = 0.0f;
    for (u32 e = 0; e < p.k; e++) {
        sum += weights[t * p.k + e] * y[((size_t)t * p.k + e) * p.width + at];
    }
    out[i] = sum;
}
