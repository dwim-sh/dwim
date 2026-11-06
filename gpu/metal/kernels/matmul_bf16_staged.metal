// out[t][r] = w[r] · x[t] for bf16 weights and a few tokens, on the matrix
// accelerators, as `matmul_ternary_staged.metal` does it for ternary
// weights: a threadgroup of two SIMD groups takes 64 rows, 32 to a SIMD
// group, over a range of the columns in steps of 32; each lane converts
// its row's next 32 weights into a stage in threadgroup memory as half
// floats, and the SIMD group multiplies the batch's activations, half
// floats laid out by `pack_halves.metal` and padded to TOKENS, into the
// stage through Metal 4's tensor operations, with the next step's weights
// loaded while the multiply runs, and two stages taken in turn. The
// columns are split among threadgroups so that there are enough to fill
// the GPU, and their partial sums `matmul_ternary_reduce.metal` adds up.
// The weights are stored by row, as the file has them.

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
// the columns of a step and of the unit the columns are split in, and the
// half floats a stage's row takes.
constant uint ROWS = 64;
constant uint SG_ROWS = 32;
#ifndef TOKENS
#define TOKENS 16
#endif
constant uint STEP = 32;
constant uint BLOCK = 128;
constant uint STRIDE = 36;
constant uint STAGE = SG_ROWS * STRIDE;

// The four bf16 weights of two words, in memory order, as half floats.
static half4 halves(uint2 pair) {
    return half4(as_type<float>(pair.x << 16), as_type<float>(pair.x & 0xffff0000u),
                 as_type<float>(pair.y << 16), as_type<float>(pair.y & 0xffff0000u));
}

kernel void matmul_bf16_staged(
    device float* out [[buffer(0)]],
    const device uint4* w [[buffer(1)]],
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
    // The split's steps.
    uint per = (blocks + p.splits - 1) / p.splits;
    uint s0 = group.y * per * (BLOCK / STEP);
    uint s1 = min(blocks, (group.y + 1) * per) * (BLOCK / STEP);
    // The row's weights, 8 to a vector.
    const device uint4* mine_w = w + row * (p.cols / 8);

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

    uint4 cur[4];
    if (s0 < s1) {
        for (uint i = 0; i < 4; i++) {
            cur[i] = mine_w[s0 * (STEP / 8) + i];
        }
    }
    for (uint s = s0; s < s1; s++) {
        uint which = (s - s0) & 1;
        threadgroup half* dst = mine + which * STAGE + lane * STRIDE;
        for (uint i = 0; i < 4; i++) {
            *(threadgroup half4*)(dst + 8 * i) = halves(cur[i].xy);
            *(threadgroup half4*)(dst + 8 * i + 4) = halves(cur[i].zw);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (s + 1 < s1) {
            for (uint i = 0; i < 4; i++) {
                cur[i] = mine_w[(s + 1) * (STEP / 8) + i];
            }
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
