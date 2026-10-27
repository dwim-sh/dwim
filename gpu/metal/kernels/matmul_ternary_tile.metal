// out[t][r] = w[r] · x[t] for ternary weights and a batch of tokens, as a
// tiled matrix product on the SIMD groups' 8x8 matrix units. A threadgroup
// takes 64 rows and TOKENS tokens, 16, 32 or 64 as the batch needs, its
// four SIMD groups 32 rows and half the tokens each. For each
// block of 128 columns it unpacks the rows' block once into threadgroup
// memory, scaled and as half floats, which hold a scaled trit exactly; the
// SIMD groups then multiply it into the tokens' activations, already half
// floats laid out by `pack_halves.metal` and padded to whole tiles, which
// they load straight from device memory, and sum the products in single
// precision. The next block's words are fetched into registers before the
// multiplies of the current one, so that their latency hides behind them.
// Two threads unpack each row, a share of its words each, by
// `trits.metal`, which the device puts ahead of this. The weights are
// stored block-major, as `matmul_ternary_sums.metal` describes.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
    uint splits;
};

// Rows and tokens of a tile, and columns of a block.
constant uint ROWS = 64;
#ifndef TOKENS
#define TOKENS 64
#endif
// 8x8 matrices of tokens and of rows a SIMD group takes.
constant uint TI = TOKENS / 16;
constant uint RJ = 4;
constant uint BLOCK = 128;
// Half floats a row of the unpacked weights takes.
constant uint STRIDE = 132;

// The words of a row the thread unpacks: the first three to the first
// thread of the row, the rest to the second, and the last, which holds the
// scale, to both.
static uint first_word(uint lid) {
    return lid >= ROWS ? 3 : 0;
}

static uint last_word(uint lid) {
    return lid >= ROWS ? WORDS : 3;
}

kernel void matmul_ternary_tile(
    device float* out [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device half* x [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // The tile's weights for the block, by row, a row's 128 columns padded
    // to STRIDE so that the threads writing the same columns of different
    // rows hit different banks.
    threadgroup half wt[ROWS * STRIDE];
    // Each SIMD group's 8x8 of results, for a tile at the matrix's edge.
    threadgroup float edge[4][64];

    uint r0 = group.x * ROWS;
    uint t0 = group.y * TOKENS;
    uint blocks = p.cols / BLOCK;
    // The SIMD group's rows and tokens of the tile.
    uint sr = (sg % 2) * 32;
    uint st = (sg / 2) * (TOKENS / 2);
    // The row the thread unpacks, clamped into the matrix; a row past the
    // end gives results that are not kept.
    uint row = lid % ROWS;
    uint wrow = min(r0 + row, p.rows - 1);

    simdgroup_float8x8 acc[TI][RJ];
    for (uint i = 0; i < TI; i++) {
        for (uint j = 0; j < RJ; j++) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    uint words[WORDS];
    for (uint j = 0; j < WORDS; j++) {
        words[j] = w[wrow * WORDS + j];
    }
    for (uint b = 0; b < blocks; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        half scale = as_type<half>(ushort(words[6] >> 16));
        for (uint j = first_word(lid); j < min(last_word(lid), 6u); j++) {
            Split s = split(words[j]);
            for (uint n = 0; n < 5; n++) {
                half4 t = peel(s) * scale;
                // Words 0 to 3 hold elements 4j + i + 16n in byte 4j + i,
                // and words 4 and 5 elements 80 + 4(j - 4) + i + 8n.
                uint e = j < 4 ? 4 * j + 16 * n : 80 + 4 * (j - 4) + 8 * n;
                *(threadgroup half4*)(wt + row * STRIDE + e) = t;
            }
        }
        if (lid >= ROWS) {
            // The low two bytes of the last word hold four trits each: byte
            // 24 + i holds elements 120 + i + 2n.
            Split s = split(words[6] & 0xffffu);
            for (uint n = 0; n < 4; n++) {
                half4 t = peel(s) * scale;
                *(threadgroup half2*)(wt + row * STRIDE + 120 + 2 * n) = t.xy;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (b + 1 < blocks) {
            const device uint* next = w + ((b + 1) * p.rows + wrow) * WORDS;
            for (uint j = first_word(lid); j < last_word(lid); j++) {
                words[j] = next[j];
            }
            words[6] = next[6];
        }
        const device half* xb = x + (t0 + st) * p.cols + b * BLOCK;
        for (uint k = 0; k < BLOCK; k += 8) {
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
