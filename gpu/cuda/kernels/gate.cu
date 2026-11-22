// A gate on a message: x = x - tanh(forget) * x + tanh(gain) * message,
// either with the gain and what is forgotten given, an element a thread,
// or with both as projections of a few dimensions, which are themselves
// projections of x and of the message, each normalized by its root mean
// square. The second is two kernels, for up to `GATES` gates at once,
// each with its own input, message and weights, and each taking eight
// tokens to a block so that the weights, which are bf16, are read once
// for the eight: `gate_down` takes rows of the projections down, a warp a
// row with its lanes eight weights of both matrices each at a time, and
// the warp of a gate's first row adds up the squares of both rows of
// activations as it goes; `gate_up` takes 256 elements of x, a thread
// each, with the rows of the two projections back up for them. Between
// the two, a token has `GATE_DOWN` floats of `down` for each gate, whole
// fours of them: what its input comes down to, what its message does, and
// what normalizes each. `gate_down_few` is `gate_down` for no more tokens
// than a block takes, a block a row: all its warps on the row, so that a
// token alone waits for a round or two of weights rather than nine.
//
// A batch on a GPU with tensor cores has the projections down from the
// tile kernels instead, as matrix multiplications of the activations'
// halves, and `gate_norms` for what normalizes each row. `down` is then
// `split`: each gate's rows of what the inputs come down to, then each
// gate's of what the messages do, then a pair of norms a row.

// Most dimensions the gate goes through, tokens to a block, and gates to
// a grid.
#define GATE_RANK 64
#define GATE_TOKENS 8
#define GATE_DOWN(rank) (2 * (rank) + 4)
#define GATES 4

struct GateJobs {
    float* x[GATES];
    const float* message[GATES];
    const uint4* down_input[GATES];
    const uint4* down_message[GATES];
    const uint4* gain[GATES];
    const uint4* forget[GATES];
    u32 n;
    u32 dim;
    u32 rank;
    float eps;
    u32 split;
};

// The four bf16 weights in two words, in memory order.
__device__ __forceinline__ float4 bf16_weights(u32 a, u32 b) {
    return make_float4(bf16_lo(a), bf16_hi(a), bf16_lo(b), bf16_hi(b));
}

template <bool FEW>
__device__ __forceinline__ void gate_down_body(float* down, GateJobs p) {
    __shared__ float partial[8][4 * GATE_TOKENS];
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    const u32 blocks = FEW ? p.rank : p.rank / 8;
    const u32 job = blockIdx.x / blocks;
    const u32 r = FEW ? blockIdx.x % blocks : (blockIdx.x % blocks) * 8 + warp;
    // The warps of a gate's first row add up its squares.
    const bool norms = r == 0;
    const u32 t0 = blockIdx.y * GATE_TOKENS;
    u32 vectors = p.dim / 4;
    u32 chunks = p.dim / 8;
    const uint4* down_input = p.down_input[job];
    const uint4* down_message = p.down_message[job];
    // A token past the last is taken as the last, and not written.
    const float4* xr[GATE_TOKENS];
    const float4* mr[GATE_TOKENS];
    float from_input[GATE_TOKENS], from_message[GATE_TOKENS];
    float squares[GATE_TOKENS], message_squares[GATE_TOKENS];
#pragma unroll
    for (u32 t = 0; t < GATE_TOKENS; t++) {
        size_t row = (size_t)min(t0 + t, p.n - 1) * vectors;
        xr[t] = (const float4*)p.x[job] + row;
        mr[t] = (const float4*)p.message[job] + row;
        from_input[t] = 0.0f;
        from_message[t] = 0.0f;
        squares[t] = 0.0f;
        message_squares[t] = 0.0f;
    }
    for (u32 c = FEW ? threadIdx.x : lane; c < chunks; c += FEW ? 256 : 32) {
        uint4 wi = down_input[(size_t)r * chunks + c];
        uint4 wm = down_message[(size_t)r * chunks + c];
        float4 wi0 = bf16_weights(wi.x, wi.y), wi1 = bf16_weights(wi.z, wi.w);
        float4 wm0 = bf16_weights(wm.x, wm.y), wm1 = bf16_weights(wm.z, wm.w);
#pragma unroll
        for (u32 t = 0; t < GATE_TOKENS; t++) {
            float4 x0 = xr[t][2 * c], x1 = xr[t][2 * c + 1];
            float4 m0 = mr[t][2 * c], m1 = mr[t][2 * c + 1];
            from_input[t] += dot4(wi0, x0) + dot4(wi1, x1);
            from_message[t] += dot4(wm0, m0) + dot4(wm1, m1);
            if (norms) {
                squares[t] += dot4(x0, x0) + dot4(x1, x1);
                message_squares[t] += dot4(m0, m0) + dot4(m1, m1);
            }
        }
    }
    float* to = down + ((size_t)job * p.n + t0) * GATE_DOWN(p.rank);
#pragma unroll
    for (u32 t = 0; t < GATE_TOKENS; t++) {
        from_input[t] = warp_sum(from_input[t]);
        from_message[t] = warp_sum(from_message[t]);
        if (norms) {
            squares[t] = warp_sum(squares[t]);
            message_squares[t] = warp_sum(message_squares[t]);
        }
        if (FEW && lane == 0) {
            partial[warp][4 * t] = from_input[t];
            partial[warp][4 * t + 1] = from_message[t];
            partial[warp][4 * t + 2] = squares[t];
            partial[warp][4 * t + 3] = message_squares[t];
        }
    }
    if (FEW) {
        // A thread a token adds up the warps' sums.
        __syncthreads();
        if (threadIdx.x >= GATE_TOKENS) {
            return;
        }
#pragma unroll
        for (u32 t = 0; t < GATE_TOKENS; t++) {
            from_input[t] = 0.0f;
            from_message[t] = 0.0f;
            squares[t] = 0.0f;
            message_squares[t] = 0.0f;
#pragma unroll
            for (u32 w = 0; w < 8; w++) {
                from_input[t] += partial[w][4 * t];
                from_message[t] += partial[w][4 * t + 1];
                squares[t] += partial[w][4 * t + 2];
                message_squares[t] += partial[w][4 * t + 3];
            }
        }
    }
#pragma unroll
    for (u32 t = 0; t < GATE_TOKENS; t++) {
        if ((FEW ? threadIdx.x == t : lane == 0) && t0 + t < p.n) {
            float* token = to + t * GATE_DOWN(p.rank);
            token[r] = from_input[t];
            token[p.rank + r] = from_message[t];
            if (norms) {
                token[2 * p.rank] = rsqrtf(squares[t] / (float)p.dim + p.eps);
                token[2 * p.rank + 1] = rsqrtf(message_squares[t] / (float)p.dim + p.eps);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gate_down(float* down, GateJobs p) {
    gate_down_body<false>(down, p);
}

extern "C" __global__ void __launch_bounds__(256) gate_down_few(float* down, GateJobs p) {
    gate_down_body<true>(down, p);
}

// What normalizes a token's row of a gate's input and of its message: a
// block a row, the tokens across the grid and the gates down it.
extern "C" __global__ void __launch_bounds__(256) gate_norms(float* norms, GateJobs p) {
    __shared__ float partial[8][2];
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    const u32 job = blockIdx.y;
    u32 vectors = p.dim / 4;
    const float4* x = (const float4*)p.x[job] + (size_t)blockIdx.x * vectors;
    const float4* message = (const float4*)p.message[job] + (size_t)blockIdx.x * vectors;
    float squares = 0.0f;
    float message_squares = 0.0f;
    for (u32 i = threadIdx.x; i < vectors; i += 256) {
        float4 a = x[i], b = message[i];
        squares += dot4(a, a);
        message_squares += dot4(b, b);
    }
    squares = warp_sum(squares);
    message_squares = warp_sum(message_squares);
    if (lane == 0) {
        partial[warp][0] = squares;
        partial[warp][1] = message_squares;
    }
    __syncthreads();
    if (threadIdx.x < 2) {
        float total = 0.0f;
#pragma unroll
        for (u32 w = 0; w < 8; w++) {
            total += partial[w][threadIdx.x];
        }
        norms[((size_t)job * p.n + blockIdx.x) * 2 + threadIdx.x] = rsqrtf(total / (float)p.dim + p.eps);
    }
}

extern "C" __global__ void __launch_bounds__(256) gate_up(const float* down, GateJobs p) {
    __shared__ float4 downs[GATE_TOKENS * GATE_RANK / 4];
    const u32 lid = threadIdx.x;
    const u32 blocks = (p.dim + 255) / 256;
    const u32 job = blockIdx.x / blocks;
    const u32 i = (blockIdx.x % blocks) * 256 + lid;
    const u32 t0 = blockIdx.y * GATE_TOKENS;
    u32 per_row = p.rank / 8;
    u32 quads = p.rank / 4;
    // The element's rows of the two projections up, read before they are
    // used: the index is clamped rather than the load skipped, so that
    // they go out together.
    uint4 wg[GATE_RANK / 8], wf[GATE_RANK / 8];
#pragma unroll
    for (u32 c = 0; c < GATE_RANK / 8; c++) {
        size_t at = (size_t)min(i, p.dim - 1) * per_row + min(c, per_row - 1);
        wg[c] = p.gain[job][at];
        wf[c] = p.forget[job][at];
    }
    // The tokens' few dimensions, for every thread: the two projections
    // down, each normalized, added.
    for (u32 at = lid; at < GATE_TOKENS * quads; at += 256) {
        u32 t = min(t0 + at / quads, p.n - 1);
        size_t row = (size_t)job * p.n + t;
        const float* from = down + row * GATE_DOWN(p.rank);
        const float* input = from;
        const float* of_message = from + p.rank;
        const float* norms = from + 2 * p.rank;
        if (p.split) {
            input = down + row * p.rank;
            of_message = input + (size_t)GATES * p.n * p.rank;
            norms = down + (size_t)2 * GATES * p.n * p.rank + row * 2;
        }
        u32 quad = (at % quads) * 4;
        downs[at] = add4(
            scale4(*(const float4*)(input + quad), norms[0]), scale4(*(const float4*)(of_message + quad), norms[1]));
    }
    __syncthreads();
    if (i >= p.dim) {
        return;
    }
    float g[GATE_TOKENS], f[GATE_TOKENS];
#pragma unroll
    for (u32 t = 0; t < GATE_TOKENS; t++) {
        g[t] = 0.0f;
        f[t] = 0.0f;
    }
#pragma unroll
    for (u32 c = 0; c < GATE_RANK / 8; c++) {
        if (c >= per_row) {
            continue;
        }
        float4 wg0 = bf16_weights(wg[c].x, wg[c].y), wg1 = bf16_weights(wg[c].z, wg[c].w);
        float4 wf0 = bf16_weights(wf[c].x, wf[c].y), wf1 = bf16_weights(wf[c].z, wf[c].w);
#pragma unroll
        for (u32 t = 0; t < GATE_TOKENS; t++) {
            float4 lo = downs[t * quads + 2 * c], hi = downs[t * quads + 2 * c + 1];
            g[t] += dot4(wg0, lo) + dot4(wg1, hi);
            f[t] += dot4(wf0, lo) + dot4(wf1, hi);
        }
    }
    float* x = p.x[job];
    const float* message = p.message[job];
#pragma unroll
    for (u32 t = 0; t < GATE_TOKENS; t++) {
        if (t0 + t < p.n) {
            size_t at = (size_t)(t0 + t) * p.dim + i;
            x[at] = x[at] - tanhf(f[t]) * x[at] + tanhf(g[t]) * message[at];
        }
    }
}
