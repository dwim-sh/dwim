// Merges the chunks of `attention.cu`: one block per (token, head), each
// thread taking one element of the head. The chunks' softmaxes were taken
// against their own maxima, so each is rescaled to the maximum over the
// chunks the token attends to before the sums are added.

extern "C" __global__ void __launch_bounds__(256) attention_combine(float* out, const float* partials, AttentionParams p) {
    const u32 lid = threadIdx.x;
    u32 t = p.first + blockIdx.x / p.n_heads;
    u32 h = blockIdx.x % p.n_heads;
    if (lid >= p.head_dim) {
        return;
    }
    u32 len = p.pos + t + 1;
    u32 chunks = (len + CHUNK - 1) / CHUNK;
    u32 stride = p.head_dim + 2;
    u32 base = ((t - p.first) * p.n_heads + h) * p.chunks * stride;
    float m = -1e30f;
    for (u32 c = 0; c < chunks; c++) {
        m = fmaxf(m, partials[base + c * stride]);
    }
    float total = 0.0f;
    float o = 0.0f;
    for (u32 c = 0; c < chunks; c++) {
        float scale = expf(partials[base + c * stride] - m);
        total += scale * partials[base + c * stride + 1];
        o += scale * partials[base + c * stride + 2 + lid];
    }
    out[(t * p.n_heads + h) * p.head_dim + lid] = o / total;
}
