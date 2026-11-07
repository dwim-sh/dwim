// Rotary position embeddings: each thread rotates one pair of elements,
// half the rotated part of a head apart, by the angle for the token's
// position.

extern "C" __global__ void rope(float* x, const float2* table, RopeParams p) {
    u32 half = p.rot_dim / 2;
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= p.n * p.n_heads * half) {
        return;
    }
    u32 i = id % half;
    // Heads are numbered across tokens: token t's first is t * n_heads.
    u32 head = id / half;
    u32 t = head / p.n_heads;
    u32 base = head * p.head_dim;
    float2 angle = table[(p.pos + t) * half + i];
    float a = x[base + i];
    float b = x[base + i + half];
    x[base + i] = a * angle.x - b * angle.y;
    x[base + i + half] = b * angle.x + a * angle.y;
}
