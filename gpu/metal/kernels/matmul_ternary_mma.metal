// out[t][r] = w[r] · x[t] for ternary weights and a batch of tokens, as a
// tiled matrix product on the GPU's matrix accelerators through Metal 4's
// tensor operations. A threadgroup takes ROWS rows and 64 tokens. For each
// block of 128 columns it unpacks the rows' block once into threadgroup
// memory, scaled and as half floats, which hold a scaled trit exactly, and
// multiplies the tokens' activations, half floats laid out by
// `pack_halves.metal`, into it, summing the products in single precision
// across the blocks in a cooperative tensor, which stores the tile with
// the matrix's and the batch's edges cut off. The next block's words are
// fetched into registers before the multiplies of the current one, so that
// their latency hides behind them. The unpacking is as in
// `matmul_ternary_tile.metal`, whose kernel is the one for a GPU without
// the accelerators.

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

// Rows and tokens of a tile, and columns of a block.
constant uint ROWS = 64;
constant uint TOKENS = 64;
constant uint BLOCK = 128;
constant uint THREADS = 128;
// Words per block: 28 bytes.
constant uint WORDS = 7;
// Half floats a row of the unpacked weights takes, padded so that the
// threads writing the same columns of different rows hit different banks.
constant uint STRIDE = 132;
// Threads that unpack a row, and the words each takes: the last, which
// holds the scale, to all of them.
constant uint PER_ROW = THREADS / ROWS;

// The four bytes of a word as two pairs in 16-bit halves, so that tripling
// them does not carry between bytes.
struct Split {
    uint even;
    uint odd;
};

static Split split(uint word) {
    return Split { word & 0x00ff00ffu, (word >> 8) & 0x00ff00ffu };
}

// Peels the most significant trit off each of the four bytes, as -1, 0, or
// 1, in the bytes' order.
static half4 peel(thread Split& s) {
    uint even = s.even * 3;
    uint odd = s.odd * 3;
    s.even = even & 0x00ff00ffu;
    s.odd = odd & 0x00ff00ffu;
    return half4(half((even >> 8) & 0xff), half((odd >> 8) & 0xff), half(even >> 24), half(odd >> 24)) - 1.0h;
}

// Unpacks words `first` to `last` of a row's block into its row of the
// unpacked weights: words 0 to 3 hold elements 4j + i + 16n in byte 4j + i,
// words 4 and 5 elements 80 + 4(j - 4) + i + 8n, and the low two bytes of
// the last word, four trits each, elements 120 + i + 2n.
static void unpack(thread const uint* words, uint first, uint last, threadgroup half* row) {
    half scale = as_type<half>(ushort(words[6] >> 16));
    for (uint j = first; j < min(last, 6u); j++) {
        Split s = split(words[j]);
        for (uint n = 0; n < 5; n++) {
            uint e = j < 4 ? 4 * j + 16 * n : 80 + 4 * (j - 4) + 8 * n;
            *(threadgroup half4*)(row + e) = peel(s) * scale;
        }
    }
    if (last == WORDS) {
        Split s = split(words[6] & 0xffffu);
        for (uint n = 0; n < 4; n++) {
            *(threadgroup half2*)(row + 120 + 2 * n) = (peel(s) * scale).xy;
        }
    }
}

kernel void matmul_ternary_mma(
    device float* out [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    device half* x [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup half wt[ROWS * STRIDE];

    uint r0 = group.x * ROWS;
    uint t0 = group.y * TOKENS;
    uint blocks = p.cols / BLOCK;
    // The row the thread unpacks, clamped into the matrix, and its share of
    // the row's words: the first threads of the rows the first words. A row
    // past the end gives results that are not kept.
    uint row = lid % ROWS;
    uint share = lid / ROWS;
    uint first = share * WORDS / PER_ROW;
    uint last = (share + 1) * WORDS / PER_ROW;
    uint wrow = min(r0 + row, p.rows - 1);
    threadgroup half* mine = wt + row * STRIDE;

    // The activations by token, the tokens padded to whole tiles; the
    // weights by row; and the results by token, as many as there are.
    auto xs = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        x, dextents<int32_t, 2>(p.cols, (p.n + TOKENS - 1) / TOKENS * TOKENS));
    auto ws = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        wt, dextents<int32_t, 2>(BLOCK, ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto results = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        out, dextents<int32_t, 2>(p.rows, p.n));
    constexpr auto desc = matmul2d_descriptor(
        TOKENS, ROWS, BLOCK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    using Xs = decltype(xs.slice<BLOCK, TOKENS>(0, 0));
    auto acc = op.get_destination_cooperative_tensor<Xs, decltype(ws), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    uint words[WORDS];
    for (uint j = 0; j < WORDS; j++) {
        words[j] = w[wrow * WORDS + j];
    }
    for (uint b = 0; b < blocks; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        unpack(words, first, last, mine);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (b + 1 < blocks) {
            const device uint* next = w + ((b + 1) * p.rows + wrow) * WORDS;
            for (uint j = first; j < last; j++) {
                words[j] = next[j];
            }
            words[6] = next[6];
        }
        auto xb = xs.slice<BLOCK, TOKENS>(b * BLOCK, t0);
        op.run(xb, ws, acc);
    }
    auto tile = results.slice(r0, t0);
    acc.store(tile);
}
