// out[t][r] = w[r] · x[t] for Q4_0 or Q8_0 weights, BITS 4 or 8, and a
// batch of tokens, as a tiled matrix product on the GPU's matrix
// accelerators through Metal 4's tensor operations, as
// `matmul_ternary_mma.metal` does for ternary weights. A threadgroup takes
// 64 rows and 64 tokens. For each 128 columns, four blocks of 32, two
// threads unpack each row's blocks once into threadgroup memory, two
// blocks each, scaled and as half floats, and the tokens' activations,
// half floats laid out by `pack_halves.metal`, are multiplied into them,
// the products summed in single precision across the steps in a
// cooperative tensor, which stores the tile with the matrix's and the
// batch's edges cut off. The next columns' quants and scales are fetched
// into registers before the multiplies of the current ones, so that their
// latency hides behind them. The unpacking is as in
// `matmul_quant_tile.metal`, whose kernel is the one for a GPU without the
// accelerators.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
    uint rows;
    uint cols;
    uint n;
    uint splits;
};

#ifndef BITS
#define BITS 4
#endif

// Rows and tokens of a tile, and the columns a step takes.
constant uint ROWS = 64;
constant uint TOKENS = 64;
constant uint COLS = 128;
constant uint THREADS = 128;
// Half floats a row of the unpacked weights takes, padded so that the
// threads writing the same columns of different rows hit different banks.
constant uint STRIDE = 132;
// Weights in a block, and the 32-bit words of quants it takes.
constant uint BLOCK = 32;
constant uint WORDS = BLOCK * BITS / 32;
// Threads that unpack a row, and the blocks of a step each takes.
constant uint PER_ROW = THREADS / ROWS;
constant uint PER_THREAD = COLS / BLOCK / PER_ROW;

// A thread's quants and scales for a step: its blocks' words as 16-byte
// loads.
struct Fetch {
    uint4 q[PER_THREAD * WORDS / 4];
    half d[PER_THREAD];
};

// The thread's blocks from `first` on.
static Fetch fetch(const device uint4* w, const device half* scales, uint first) {
    Fetch f;
    for (uint k = 0; k < PER_THREAD; k++) {
        for (uint v = 0; v < WORDS / 4; v++) {
            f.q[k * (WORDS / 4) + v] = w[(first + k) * (WORDS / 4) + v];
        }
        f.d[k] = scales[first + k];
    }
    return f;
}

#if BITS == 4
// The low nibbles of a word's four bytes, less eight, in memory order, and
// the high nibbles likewise.
static half4 low(uint word) {
    return half4(half(word & 0xfu), half((word >> 8) & 0xfu), half((word >> 16) & 0xfu), half((word >> 24) & 0xfu)) - 8.0h;
}

static half4 high(uint word) {
    return half4(half((word >> 4) & 0xfu), half((word >> 12) & 0xfu), half((word >> 20) & 0xfu), half(word >> 28)) - 8.0h;
}

// Unpacks a block into 32 half floats: byte j holds element j in its low
// nibble and element j + 16 in its high one.
static void unpack_block(thread const uint4* q, half d, threadgroup half* out) {
    uint words[4] = { q[0].x, q[0].y, q[0].z, q[0].w };
    for (uint v = 0; v < 4; v++) {
        *(threadgroup half4*)(out + 4 * v) = low(words[v]) * d;
        *(threadgroup half4*)(out + 16 + 4 * v) = high(words[v]) * d;
    }
}
#else
// Unpacks a block into 32 half floats: its bytes are its elements in order
// as signed integers.
static void unpack_block(thread const uint4* q, half d, threadgroup half* out) {
    uint words[8] = { q[0].x, q[0].y, q[0].z, q[0].w, q[1].x, q[1].y, q[1].z, q[1].w };
    for (uint v = 0; v < 8; v++) {
        *(threadgroup half4*)(out + 4 * v) = half4(as_type<char4>(words[v])) * d;
    }
}
#endif

// Unpacks the thread's blocks into its part of its row.
static void unpack(thread const Fetch& f, threadgroup half* mine) {
    for (uint k = 0; k < PER_THREAD; k++) {
        unpack_block(f.q + k * (WORDS / 4), f.d[k], mine + k * BLOCK);
    }
}

kernel void matmul_quant_mma(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    device half* x [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup half wt[ROWS * STRIDE];

    uint r0 = group.x * ROWS;
    uint t0 = group.y * TOKENS;
    uint steps = p.cols / COLS;
    uint blocks = p.cols / BLOCK;
    // The row the thread unpacks, clamped into the matrix, and its share
    // of the row: the threads of a row are adjacent, so that they read the
    // row's blocks of a step together. A row past the end gives results
    // that are not kept.
    uint row = lid / PER_ROW;
    uint share = lid % PER_ROW;
    uint wrow = min(r0 + row, p.rows - 1);
    // The thread's first block of the row in the first step.
    uint first = wrow * blocks + share * PER_THREAD;
    threadgroup half* mine = wt + row * STRIDE + share * PER_THREAD * BLOCK;

    // The activations by token, the tokens padded to whole tiles; the
    // weights by row; and the results by token, as many as there are.
    auto xs = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        x, dextents<int32_t, 2>(p.cols, (p.n + TOKENS - 1) / TOKENS * TOKENS));
    auto ws = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        wt, dextents<int32_t, 2>(COLS, ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto results = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        out, dextents<int32_t, 2>(p.rows, p.n));
    constexpr auto desc = matmul2d_descriptor(
        TOKENS, ROWS, COLS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    using Xs = decltype(xs.slice<COLS, TOKENS>(0, 0));
    auto acc = op.get_destination_cooperative_tensor<Xs, decltype(ws), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    Fetch f = fetch(w, scales, first);
    for (uint s = 0; s < steps; s++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        unpack(f, mine);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s + 1 < steps) {
            f = fetch(w, scales, first + (s + 1) * (COLS / BLOCK));
        }
        auto xb = xs.slice<COLS, TOKENS>(s * COLS, t0);
        op.run(xb, ws, acc);
    }
    auto tile = results.slice(r0, t0);
    acc.store(tile);
}
