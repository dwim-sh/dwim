// out[t][r] = w[r] · x[t] for ternary weights and a few tokens, on the
// matrix accelerators, after the staged decode tile of Splash
// (third_party/splash, runtime/metal/kernels/shared/gguf_linear.metal): a
// threadgroup of two SIMD groups takes 64 rows, 32 to a SIMD group, over
// a range of the blocks; each lane unpacks its row's next 32 columns into
// a stage in threadgroup memory, scaled and as half floats, and the SIMD
// group multiplies the batch's activations, half floats laid out by
// `pack_halves.metal` and padded to TOKENS, into the stage through Metal
// 4's tensor operations, with the next block's words loaded while the
// multiplies run, and two stages taken in turn. The blocks are split among
// threadgroups so that there are enough to fill the GPU, and their partial
// sums `matmul_ternary_reduce.metal` adds up. The weights are stored
// block-major, as `matmul_ternary_sums.metal` describes, and unpacked by
// `trits.metal`, which the device puts ahead of this: a block's bytes hold
// the trits of columns far apart, so a step peels from each word the trits
// of the columns it covers, keeping the rest for the steps after. A stage's
// rows are padded so that the lanes writing the same columns of different
// rows hit different banks.

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

// Rows a threadgroup and a SIMD group take, the tokens of the batch's tile,
// the columns of a block and of a step, and the half floats a stage's row
// takes.
constant uint ROWS = 64;
constant uint SG_ROWS = 32;
#ifndef TOKENS
#define TOKENS 16
#endif
constant uint BLOCK = 128;
constant uint STEP = 32;
constant uint STRIDE = 36;
constant uint STAGE = SG_ROWS * STRIDE;

// A row's block being unpacked: the split words, with the trits not yet
// peeled, and the scale.
struct Block {
    Split s[7];
    half scale;
};

static Block begin(thread const uint* words) {
    Block b;
    for (uint j = 0; j < 6; j++) {
        b.s[j] = split(words[j]);
    }
    b.s[6] = split(words[6] & 0xffffu);
    b.scale = as_type<half>(ushort(words[6] >> 16));
    return b;
}

// Unpacks the columns of step `step` of a block into `row`: words 0 to 3
// hold elements 4j + i + 16n in byte 4j + i, words 4 and 5 elements
// 80 + 4(j - 4) + i + 8n, and the low two bytes of the last word, four
// trits each, elements 120 + i + 2n.
static void unpack(thread Block& b, uint step, threadgroup half* row) {
    switch (step) {
    case 0:
        for (uint j = 0; j < 4; j++) {
            *(threadgroup half4*)(row + 4 * j) = peel(b.s[j]) * b.scale;
            *(threadgroup half4*)(row + 16 + 4 * j) = peel(b.s[j]) * b.scale;
        }
        break;
    case 1:
        for (uint j = 0; j < 4; j++) {
            *(threadgroup half4*)(row + 4 * j) = peel(b.s[j]) * b.scale;
            *(threadgroup half4*)(row + 16 + 4 * j) = peel(b.s[j]) * b.scale;
        }
        break;
    case 2:
        for (uint j = 0; j < 4; j++) {
            *(threadgroup half4*)(row + 4 * j) = peel(b.s[j]) * b.scale;
        }
        for (uint j = 4; j < 6; j++) {
            *(threadgroup half4*)(row + 16 + 4 * (j - 4)) = peel(b.s[j]) * b.scale;
            *(threadgroup half4*)(row + 24 + 4 * (j - 4)) = peel(b.s[j]) * b.scale;
        }
        break;
    default:
        for (uint j = 4; j < 6; j++) {
            *(threadgroup half4*)(row + 4 * (j - 4)) = peel(b.s[j]) * b.scale;
            *(threadgroup half4*)(row + 8 + 4 * (j - 4)) = peel(b.s[j]) * b.scale;
            *(threadgroup half4*)(row + 16 + 4 * (j - 4)) = peel(b.s[j]) * b.scale;
        }
        for (uint n = 0; n < 4; n++) {
            *(threadgroup half2*)(row + 24 + 2 * n) = (peel(b.s[6]) * b.scale).xy;
        }
        break;
    }
}

kernel void matmul_ternary_staged(
    device float* out [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    device half* x [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup half stages[2 * 2 * STAGE];
    threadgroup half* mine = stages + sg * 2 * STAGE;

    uint blocks = p.cols / BLOCK;
    uint r0 = group.x * ROWS + sg * SG_ROWS;
    // The lane's row, clamped into the matrix; a row past the end gives
    // results that are not kept.
    uint row = min(r0 + lane, p.rows - 1);
    // The split's blocks.
    uint per = (blocks + p.splits - 1) / p.splits;
    uint b0 = group.y * per;
    uint b1 = min(blocks, b0 + per);

    auto xs = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        x, dextents<int32_t, 2>(p.cols, TOKENS));
    auto s0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        mine, dextents<int32_t, 2>(STEP, SG_ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto s1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        mine + STAGE, dextents<int32_t, 2>(STEP, SG_ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    constexpr auto desc = matmul2d_descriptor(
        TOKENS, SG_ROWS, STEP, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<1>> op;
    using Xs = decltype(xs.slice<STEP, TOKENS>(0, 0));
    auto acc = op.get_destination_cooperative_tensor<Xs, decltype(s0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    uint words[WORDS];
    if (b0 < b1) {
        const device uint* base = w + (b0 * p.rows + row) * WORDS;
        for (uint j = 0; j < WORDS; j++) {
            words[j] = base[j];
        }
    }
    uint which = 0;
    for (uint b = b0; b < b1; b++) {
        Block block = begin(words);
        if (b + 1 < b1) {
            const device uint* next = w + ((b + 1) * p.rows + row) * WORDS;
            for (uint j = 0; j < WORDS; j++) {
                words[j] = next[j];
            }
        }
        for (uint step = 0; step < BLOCK / STEP; step++) {
            unpack(block, step, mine + which * STAGE + lane * STRIDE);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto xb = xs.slice<STEP, TOKENS>(int32_t(b * BLOCK + step * STEP), 0);
            if (which) {
                op.run(xb, s1, acc);
            } else {
                op.run(xb, s0, acc);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            which ^= 1;
        }
    }
    // The results by token, as many as there are: to the output, or to the
    // split's partial sums. A SIMD group whose rows all lie past the
    // matrix stores nothing: a store from past the tensor's edge is not
    // cut off at it.
    if (r0 >= p.rows) {
        return;
    }
    device float* dst = p.splits == 1 ? out : partial + group.y * p.n * p.rows;
    auto results = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        dst, dextents<int32_t, 2>(p.rows, p.n));
    auto tile = results.slice(r0, 0);
    acc.store(tile);
}
