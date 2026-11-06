// out[t][r] = w[r] · x[t] for weights four bits each and a few tokens, on
// the matrix accelerators, as `matmul_bf16_staged.metal` does it for bf16
// weights: a threadgroup of two SIMD groups takes 64 rows, 32 to a SIMD
// group, over a range of the columns in steps of 32; each lane unpacks its
// row's next 32 codes, a chunk of 16 bytes, by their group's scale and
// bias into a stage in threadgroup memory as half floats, and the SIMD
// group multiplies the batch's activations, half floats laid out by
// `pack_halves.metal` and padded to TOKENS, into the stage through Metal
// 4's tensor operations, with the next step's chunk loaded while the
// multiply runs, and two stages taken in turn. The columns are split
// among threadgroups so that there are enough to fill the GPU, and their
// partial sums `matmul_ternary_reduce.metal` adds up. The weights are
// stored by row as `affine4.rs` packs them.

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

constant uint ROWS = 64;
constant uint SG_ROWS = 32;
#ifndef TOKENS
#define TOKENS 16
#endif
constant uint STEP = 32;
constant uint BLOCK = 128;
constant uint STRIDE = 36;
constant uint STAGE = SG_ROWS * STRIDE;

// The eight codes of a word, in memory order, as the weights they stand
// for.
static void unpack(uint word, float scale, float bias, threadgroup half* dst) {
    float4 lo = float4(word & 15, (word >> 4) & 15, (word >> 8) & 15, (word >> 12) & 15);
    float4 hi = float4((word >> 16) & 15, (word >> 20) & 15, (word >> 24) & 15, word >> 28);
    *(threadgroup half4*)dst = half4(lo * scale + bias);
    *(threadgroup half4*)(dst + 4) = half4(hi * scale + bias);
}

kernel void matmul_affine4_staged(
    device float* out [[buffer(0)]],
    const device uchar* w [[buffer(1)]],
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
    uint row = min(r0 + lane, p.rows - 1);
    uint per = (blocks + p.splits - 1) / p.splits;
    uint s0 = group.y * per * (BLOCK / STEP);
    uint s1 = min(blocks, (group.y + 1) * per) * (BLOCK / STEP);
    uint row_bytes = p.cols / 2 + p.cols / 16;
    const device uint4* chunks = (const device uint4*)(w + row * row_bytes);
    const device half2* groups = (const device half2*)(w + row * row_bytes + p.cols / 2);

    auto xs = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        x, dextents<int32_t, 2>(p.cols, TOKENS));
    auto st0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        mine, dextents<int32_t, 2>(STEP, SG_ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    auto st1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        mine + STAGE, dextents<int32_t, 2>(STEP, SG_ROWS), array<int32_t, 2>{1, int32_t(STRIDE)});
    constexpr auto desc = matmul2d_descriptor(
        TOKENS, SG_ROWS, STEP, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<1>> op;
    using Xs = decltype(xs.slice<STEP, TOKENS>(0, 0));
    auto acc = op.get_destination_cooperative_tensor<Xs, decltype(st0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    uint4 cur = 0;
    half2 sb = 0.0h;
    if (s0 < s1) {
        cur = chunks[s0];
        sb = groups[s0 / 2];
    }
    for (uint s = s0; s < s1; s++) {
        uint which = (s - s0) & 1;
        threadgroup half* dst = mine + which * STAGE + lane * STRIDE;
        for (uint k = 0; k < 4; k++) {
            unpack(cur[k], float(sb.x), float(sb.y), dst + 8 * k);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (s + 1 < s1) {
            cur = chunks[s + 1];
            sb = groups[(s + 1) / 2];
        }
        auto xb = xs.slice<STEP, TOKENS>(int32_t(s * STEP), 0);
        if (which) {
            op.run(xb, st1, acc);
        } else {
            op.run(xb, st0, acc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 >= p.rows) {
        return;
    }
    device float* dst = p.splits == 1 ? out : partial + group.y * p.n * p.rows;
    auto results = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        dst, dextents<int32_t, 2>(p.rows, p.n));
    auto tile = results.slice(r0, 0);
    acc.store(tile);
}
