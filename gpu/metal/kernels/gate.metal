// A gate on a message: x = x - tanh(forget) * x + tanh(gain) * message,
// where the gain and what is forgotten are projections of a few
// dimensions, which are themselves projections of x and of the message,
// each normalized by its root mean square. One threadgroup per token: its
// threads add up the squares of both rows; each SIMD group then takes
// rows of the projections down in turn, its lanes eight weights of both
// matrices each at a time; and each thread takes elements of x, with the
// rows of the two projections back up for them. The weights are bf16.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    uint rank;
    float eps;
};

constant uint THREADS = 256;
// Most dimensions the gate goes through.
constant uint RANK = 64;

// The four bf16 weights in two words, in memory order: the low half of a
// word first, as little-endian does.
static float4 weights(uint2 pairs) {
    return float4(as_type<float>(pairs.x << 16), as_type<float>(pairs.x & 0xffff0000u),
                  as_type<float>(pairs.y << 16), as_type<float>(pairs.y & 0xffff0000u));
}

// tanh, which the GPU's own makes not a number of where its exponentials
// overflow: past 15 either way it is one to the last bit.
static float safe_tanh(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

kernel void gate(
    device float4* x [[buffer(0)]],
    const device float4* message [[buffer(1)]],
    const device uint4* down_input [[buffer(2)]],
    const device uint4* down_message [[buffer(3)]],
    const device uint4* gain [[buffer(4)]],
    const device uint4* forget [[buffer(5)]],
    constant Params& p [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    threadgroup float partial[32][2];
    threadgroup float down[RANK];
    uint vectors = p.dim / 4;
    device float4* xr = x + row * vectors;
    const device float4* mr = message + row * vectors;

    // The squares of both rows, added up across the group.
    float squares = 0.0f;
    float message_squares = 0.0f;
    for (uint i = lid; i < vectors; i += THREADS) {
        squares += dot(xr[i], xr[i]);
        message_squares += dot(mr[i], mr[i]);
    }
    squares = simd_sum(squares);
    message_squares = simd_sum(message_squares);
    if (lane == 0) {
        partial[sid][0] = squares;
        partial[sid][1] = message_squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    squares = 0.0f;
    message_squares = 0.0f;
    for (uint i = 0; i < nsg; i++) {
        squares += partial[i][0];
        message_squares += partial[i][1];
    }
    float norm = rsqrt(squares / float(p.dim) + p.eps);
    float message_norm = rsqrt(message_squares / float(p.dim) + p.eps);

    // Down to the gate's few dimensions: a row of each projection a SIMD
    // group at a time, eight weights a lane.
    uint chunks = p.dim / 8;
    for (uint r = sid; r < p.rank; r += nsg) {
        float from_input = 0.0f;
        float from_message = 0.0f;
        for (uint c = lane; c < chunks; c += width) {
            uint4 wi = down_input[r * chunks + c];
            uint4 wm = down_message[r * chunks + c];
            from_input += dot(weights(wi.xy), xr[2 * c]) + dot(weights(wi.zw), xr[2 * c + 1]);
            from_message += dot(weights(wm.xy), mr[2 * c]) + dot(weights(wm.zw), mr[2 * c + 1]);
        }
        float sum = simd_sum(from_input) * norm + simd_sum(from_message) * message_norm;
        if (lane == 0) {
            down[r] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Back up, an element of x a thread at a time: its rows of the two
    // projections against the few dimensions.
    uint per_row = p.rank / 8;
    for (uint i = lid; i < p.dim; i += THREADS) {
        float g = 0.0f;
        float f = 0.0f;
        for (uint c = 0; c < per_row; c++) {
            uint4 wg = gain[i * per_row + c];
            uint4 wf = forget[i * per_row + c];
            float4 lo = float4(down[8 * c], down[8 * c + 1], down[8 * c + 2], down[8 * c + 3]);
            float4 hi = float4(down[8 * c + 4], down[8 * c + 5], down[8 * c + 6], down[8 * c + 7]);
            g += dot(weights(wg.xy), lo) + dot(weights(wg.zw), hi);
            f += dot(weights(wf.xy), lo) + dot(weights(wf.zw), hi);
        }
        device float* xs = (device float*)xr;
        const device float* ms = (const device float*)mr;
        xs[i] = xs[i] - safe_tanh(f) * xs[i] + safe_tanh(g) * ms[i];
    }
}
