// Attention over the depth for up to M queries of the same states at once,
// with optional fused RMS normalization on each output. One threadgroup
// per token: the first pass scores every state against all M queries in
// one sweep; the second forms all M mixes, keeping them in registers when
// normalizing so no extra pass over device memory is needed.

#include <metal_stdlib>
using namespace metal;

#ifndef M
#define M 1
#endif

constant uint THREADS = 256;
// Vectors a thread takes: rows of up to 4096 elements.
constant uint PER_THREAD = 4;
constant uint STATES = 6;

struct Params {
    uint dim;
    uint states;
    uint mixes;
    uint normed;
    float eps;
    float norm_eps;
};

static device float4* pick_out(
    device float4* a, device float4* b, device float4* c, device float4* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

static const device float4* pick_query(
    const device float4* a, const device float4* b, const device float4* c, const device float4* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

static float4 load_state(
    const device float4* s0, const device float4* s1, const device float4* s2,
    const device float4* s3, const device float4* s4, const device float4* s5,
    uint s, uint at)
{
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
    device float4* out0 [[buffer(0)]],
    device float4* out1 [[buffer(1)]],
    device float4* out2 [[buffer(2)]],
    device float4* out3 [[buffer(3)]],
    const device float4* q0 [[buffer(4)]],
    const device float4* q1 [[buffer(5)]],
    const device float4* q2 [[buffer(6)]],
    const device float4* q3 [[buffer(7)]],
    const device float4* n0 [[buffer(8)]],
    const device float4* n1 [[buffer(9)]],
    const device float4* n2 [[buffer(10)]],
    const device float4* n3 [[buffer(11)]],
    const device float4* s0 [[buffer(12)]],
    const device float4* s1 [[buffer(13)]],
    const device float4* s2 [[buffer(14)]],
    const device float4* s3 [[buffer(15)]],
    const device float4* s4 [[buffer(16)]],
    const device float4* s5 [[buffer(17)]],
    constant Params& p [[buffer(18)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32][STATES * (1 + M)];
    threadgroup float after[32][M];
    uint vectors = p.dim / 4;
    uint base = row * vectors;

    for (uint s = 0; s < p.states; s++) {
        float squares = 0.0f;
        float products[M];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            products[j] = 0.0f;
        }
        #pragma clang loop unroll(full)
        for (uint k = 0; k < PER_THREAD; k++) {
            uint i = lid + k * THREADS;
            if (i < vectors) {
                float4 v = load_state(s0, s1, s2, s3, s4, s5, s, base + i);
                squares += dot(v, v);
                #pragma clang loop unroll(full)
                for (uint j = 0; j < M; j++) {
                    if (j < p.mixes) {
                        products[j] += dot(v, pick_query(q0, q1, q2, q3, j)[i]);
                    }
                }
            }
        }
        squares = simd_sum(squares);
        if (lane == 0) {
            partial[sid][s * (1 + M)] = squares;
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            float prod = simd_sum(products[j]);
            if (lane == 0) {
                partial[sid][s * (1 + M) + 1 + j] = prod;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // What normalizes each state, and each mix's normalized shares.
    float norms[STATES];
    for (uint s = 0; s < p.states; s++) {
        float squares = 0.0f;
        for (uint i = 0; i < nsg; i++) {
            squares += partial[i][s * (1 + M)];
        }
        norms[s] = rsqrt(squares / float(p.dim) + p.eps);
    }
    float shares[M][STATES];
    float inv_sqrt_dim = rsqrt(float(p.dim));
    #pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        float most = -1e30f;
        for (uint s = 0; s < p.states; s++) {
            float product = 0.0f;
            for (uint i = 0; i < nsg; i++) {
                product += partial[i][s * (1 + M) + 1 + j];
            }
            shares[j][s] = product * norms[s] * inv_sqrt_dim;
            most = max(most, shares[j][s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < p.states; s++) {
            shares[j][s] = exp(shares[j][s] - most);
            sum += shares[j][s];
        }
        for (uint s = 0; s < p.states; s++) {
            shares[j][s] = shares[j][s] / sum * norms[s];
        }
    }

    if (p.normed == 0) {
        #pragma clang loop unroll(full)
        for (uint k = 0; k < PER_THREAD; k++) {
            uint i = lid + k * THREADS;
            if (i < vectors) {
                float4 o[M];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < M; j++) {
                    o[j] = float4(0.0f);
                }
                for (uint s = 0; s < p.states; s++) {
                    float4 v = load_state(s0, s1, s2, s3, s4, s5, s, base + i);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < M; j++) {
                        o[j] += shares[j][s] * v;
                    }
                }
                #pragma clang loop unroll(full)
                for (uint j = 0; j < M; j++) {
                    if (j < p.mixes) {
                        pick_out(out0, out1, out2, out3, j)[base + i] = o[j];
                    }
                }
            }
        }
        return;
    }

    // Form all M mixes in registers, reduce their sums of squares, and write
    // them scaled by their RMS norm weights.
    float4 o[M][PER_THREAD];
    float ss[M];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        ss[j] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < PER_THREAD; k++) {
            o[j][k] = float4(0.0f);
        }
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = lid + k * THREADS;
        if (i < vectors) {
            for (uint s = 0; s < p.states; s++) {
                float4 v = load_state(s0, s1, s2, s3, s4, s5, s, base + i);
                #pragma clang loop unroll(full)
                for (uint j = 0; j < M; j++) {
                    o[j][k] += shares[j][s] * v;
                }
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < M; j++) {
                ss[j] += dot(o[j][k], o[j][k]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        float s = simd_sum(ss[j]);
        if (lane == 0) {
            after[sid][j] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        if (j < p.mixes) {
            float total = 0.0f;
            for (uint i = 0; i < nsg; i++) {
                total += after[i][j];
            }
            float scale = rsqrt(total / float(p.dim) + p.norm_eps);
            device float4* dst = pick_out(out0, out1, out2, out3, j);
            const device float4* nw = pick_query(n0, n1, n2, n3, j);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < PER_THREAD; k++) {
                uint i = lid + k * THREADS;
                if (i < vectors) {
                    dst[base + i] = o[j][k] * (scale * nw[i]);
                }
            }
        }
    }
}
