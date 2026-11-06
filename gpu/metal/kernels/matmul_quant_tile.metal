// out[t][r] = w[r] · x[t] for Q4_0 or Q8_0 weights, BITS 4 or 8, and a
// batch of tokens, as a tiled matrix product on the SIMD groups' 8x8
// matrix units, as `matmul_ternary_tile.metal` does for ternary weights. A
// threadgroup takes 64 rows and TOKENS tokens, 16, 32 or 64 as the batch
// needs, its four SIMD groups 32 rows and half the tokens each. For each
// 128 columns, four blocks of 32, two threads unpack each row's blocks
// once into threadgroup memory, two blocks each, scaled and as half
// floats: a weight times its block's f16 scale rounds to a half at the
// precision of the activations, themselves half floats laid out by
// `pack_halves.metal` and padded to whole tiles, which the SIMD groups load
// straight from device memory and multiply into the weights, summing the
// products in single precision. The next columns' quants and scales are
// fetched into registers before the multiplies of the current ones, so
// that their latency hides behind them. The quants and the scales are laid
// out by row, as `quant.rs` says.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
    uint splits;
};

#ifndef BITS
#define BITS 4
#endif
#ifndef TOKENS
#define TOKENS 64
#endif

// Rows and tokens of a tile, and the columns a step takes.
constant uint ROWS = 64;
constant uint COLS = 128;
// 8x8 matrices of tokens and of rows a SIMD group takes.
constant uint TI = TOKENS / 16;
constant uint RJ = 4;
// Half floats a row of the unpacked weights takes, padded so that the
// threads writing the same columns of different rows hit different banks.
constant uint STRIDE = 132;
// Weights in a block, and the 32-bit words of quants it takes.
constant uint BLOCK = 32;
constant uint WORDS = BLOCK * BITS / 32;
// Threads that unpack a row, and the blocks of a step each takes.
constant uint PER_ROW = 2;
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

kernel void matmul_quant_tile(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device half* x [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // The tile's weights for the step, by row.
    threadgroup half wt[ROWS * STRIDE];
    // Each SIMD group's 8x8 of results, for a tile at the matrix's edge.
    threadgroup float edge[4][64];

    uint r0 = group.x * ROWS;
    uint t0 = group.y * TOKENS;
    uint steps = p.cols / COLS;
    uint blocks = p.cols / BLOCK;
    // The SIMD group's rows and tokens of the tile.
    uint sr = (sg % 2) * 32;
    uint st = (sg / 2) * (TOKENS / 2);
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

    simdgroup_float8x8 acc[TI][RJ];
    for (uint i = 0; i < TI; i++) {
        for (uint j = 0; j < RJ; j++) {
            acc[i][j] = simdgroup_float8x8(0.0f);
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
        const device half* xb = x + (t0 + st) * p.cols + s * COLS;
        for (uint k = 0; k < COLS; k += 8) {
            simdgroup_half8x8 a[TI];
            simdgroup_half8x8 m[RJ];
            for (uint i = 0; i < TI; i++) {
                simdgroup_load(a[i], xb + 8 * i * p.cols + k, p.cols);
            }
            for (uint j = 0; j < RJ; j++) {
                simdgroup_load(m[j], wt + (sr + 8 * j) * STRIDE + k, STRIDE, ulong2(0, 0), true);
            }
            for (uint i = 0; i < TI; i++) {
                for (uint j = 0; j < RJ; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], m[j], acc[i][j]);
                }
            }
        }
    }
    // A whole tile goes straight to the output; one at the edge of the
    // matrix or the batch through threadgroup memory, a lane taking two of
    // each 8x8.
    bool whole = r0 + ROWS <= p.rows && t0 + TOKENS <= p.n;
    for (uint i = 0; i < TI; i++) {
        for (uint j = 0; j < RJ; j++) {
            uint t = t0 + st + 8 * i;
            uint r = r0 + sr + 8 * j;
            if (whole) {
                simdgroup_store(acc[i][j], out + t * p.rows + r, p.rows);
                continue;
            }
            simdgroup_store(acc[i][j], edge[sg], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint tt = t + e / 8;
                uint rr = r + e % 8;
                if (tt < p.n && rr < p.rows) {
                    out[tt * p.rows + rr] = edge[sg][e];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
