// Attention over the depth: out[t] is a mix of token t's rows of up to
// STATES states, each normalized by its root mean square, weighted by the
// softmax over the states of each normalized row's dot product with the
// query, over the square root of the width. One threadgroup per token:
// its threads take four elements each of every state, the sums of squares
// and the dot products are added up across the group, and each thread
// writes its elements of the mix.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    uint count;
    float eps;
};

constant uint THREADS = 256;
// Vectors a thread takes: rows of up to 4096 elements.
constant uint PER_THREAD = 4;
constant uint STATES = 6;

// Vector `at` of state s.
static float4 load(const device float4* s0, const device float4* s1, const device float4* s2,
                   const device float4* s3, const device float4* s4, const device float4* s5,
                   uint s, uint at) {
    switch (s) {
        case 0: return s0[at];
        case 1: return s1[at];
        case 2: return s2[at];
        case 3: return s3[at];
        case 4: return s4[at];
        default: return s5[at];
    }
}

kernel void mix(
    device float4* out [[buffer(0)]],
    const device float4* query [[buffer(1)]],
    const device float4* s0 [[buffer(2)]],
    const device float4* s1 [[buffer(3)]],
    const device float4* s2 [[buffer(4)]],
    const device float4* s3 [[buffer(5)]],
    const device float4* s4 [[buffer(6)]],
    const device float4* s5 [[buffer(7)]],
    constant Params& p [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32][2 * STATES];
    uint vectors = p.dim / 4;
    for (uint s = 0; s < p.count; s++) {
        float squares = 0.0f;
        float product = 0.0f;
        for (uint k = 0; k < PER_THREAD; k++) {
            uint i = lid + k * THREADS;
            if (i < vectors) {
                float4 v = load(s0, s1, s2, s3, s4, s5, s, row * vectors + i);
                squares += dot(v, v);
                product += dot(v, query[i]);
            }
        }
        squares = simd_sum(squares);
        product = simd_sum(product);
        if (lane == 0) {
            partial[sid][2 * s] = squares;
            partial[sid][2 * s + 1] = product;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // What normalizes each state, and its score against the query.
    float norms[STATES];
    float scores[STATES];
    float most = -1e30f;
    for (uint s = 0; s < p.count; s++) {
        float squares = 0.0f;
        float product = 0.0f;
        for (uint i = 0; i < nsg; i++) {
            squares += partial[i][2 * s];
            product += partial[i][2 * s + 1];
        }
        norms[s] = rsqrt(squares / float(p.dim) + p.eps);
        scores[s] = product * norms[s] * rsqrt(float(p.dim));
        most = max(most, scores[s]);
    }
    float sum = 0.0f;
    for (uint s = 0; s < p.count; s++) {
        scores[s] = exp(scores[s] - most);
        sum += scores[s];
    }
    // Each state's share of the mix, times what normalizes it.
    float share[STATES];
    for (uint s = 0; s < p.count; s++) {
        share[s] = scores[s] / sum * norms[s];
    }
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = lid + k * THREADS;
        if (i < vectors) {
            float4 o = float4(0.0f);
            for (uint s = 0; s < p.count; s++) {
                o += share[s] * load(s0, s1, s2, s3, s4, s5, s, row * vectors + i);
            }
            out[row * vectors + i] = o;
        }
    }
}
