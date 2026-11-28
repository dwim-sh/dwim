// First half of a message gate for up to GATES gates in one grid: projects
// each token's input x and message down through bf16 matrices down_input and
// down_message, and computes each row's inverse RMS norm on row 0.
//
// With FEW != 0 (n <= GATE_TOKENS), one threadgroup takes one rank row of
// one gate so that a single token fans out across all GPU cores; otherwise
// each SIMD group takes one rank row for up to GATE_TOKENS tokens at once
// so weights are read once per eight tokens.

#include <metal_stdlib>
using namespace metal;

#ifndef FEW
#define FEW 0
#endif

constant uint THREADS = 256;
constant uint GATE_TOKENS = 8;

struct Params {
    uint n;
    uint dim;
    uint rank;
    float eps;
};

static const device float4* pick_f4(
    const device float4* a, const device float4* b, const device float4* c, const device float4* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

static const device uint4* pick_u4(
    const device uint4* a, const device uint4* b, const device uint4* c, const device uint4* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

// The four bf16 weights in two words, in memory order.
static float4 weights(uint2 pairs) {
    return float4(as_type<float>(pairs.x << 16), as_type<float>(pairs.x & 0xffff0000u),
                  as_type<float>(pairs.y << 16), as_type<float>(pairs.y & 0xffff0000u));
}

kernel void gate_down(
    device float* down [[buffer(0)]],
    const device float4* x0 [[buffer(1)]],
    const device float4* x1 [[buffer(2)]],
    const device float4* x2 [[buffer(3)]],
    const device float4* x3 [[buffer(4)]],
    const device float4* m0 [[buffer(5)]],
    const device float4* m1 [[buffer(6)]],
    const device float4* m2 [[buffer(7)]],
    const device float4* m3 [[buffer(8)]],
    const device uint4* di0 [[buffer(9)]],
    const device uint4* di1 [[buffer(10)]],
    const device uint4* di2 [[buffer(11)]],
    const device uint4* di3 [[buffer(12)]],
    const device uint4* dm0 [[buffer(13)]],
    const device uint4* dm1 [[buffer(14)]],
    const device uint4* dm2 [[buffer(15)]],
    const device uint4* dm3 [[buffer(16)]],
    constant Params& p [[buffer(17)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    threadgroup float partial[32][4 * GATE_TOKENS];
    uint blocks = FEW ? p.rank : p.rank / nsg;
    uint job = wg.x / blocks;
    uint r = FEW ? wg.x % blocks : (wg.x % blocks) * nsg + sid;
    bool norms = r == 0;
    uint t0 = wg.y * GATE_TOKENS;
    uint count = min(GATE_TOKENS, p.n - t0);
    uint vectors = p.dim / 4;
    uint chunks = p.dim / 8;

    const device float4* x = pick_f4(x0, x1, x2, x3, job);
    const device float4* m = pick_f4(m0, m1, m2, m3, job);
    const device uint4* down_input = pick_u4(di0, di1, di2, di3, job);
    const device uint4* down_message = pick_u4(dm0, dm1, dm2, dm3, job);

    float from_input[GATE_TOKENS] = {};
    float from_message[GATE_TOKENS] = {};
    float squares[GATE_TOKENS] = {};
    float message_squares[GATE_TOKENS] = {};

    uint start = FEW ? lid : lane;
    uint step = FEW ? THREADS : width;
    for (uint c = start; c < chunks; c += step) {
        uint4 wi = down_input[r * chunks + c];
        uint4 wm = down_message[r * chunks + c];
        float4 wi0 = weights(wi.xy);
        float4 wi1 = weights(wi.zw);
        float4 wm0 = weights(wm.xy);
        float4 wm1 = weights(wm.zw);
        for (uint t = 0; t < count; t++) {
            const device float4* xr = x + (t0 + t) * vectors;
            const device float4* mr = m + (t0 + t) * vectors;
            float4 xv0 = xr[2 * c];
            float4 xv1 = xr[2 * c + 1];
            float4 mv0 = mr[2 * c];
            float4 mv1 = mr[2 * c + 1];
            from_input[t] += dot(wi0, xv0) + dot(wi1, xv1);
            from_message[t] += dot(wm0, mv0) + dot(wm1, mv1);
            if (norms) {
                squares[t] += dot(xv0, xv0) + dot(xv1, xv1);
                message_squares[t] += dot(mv0, mv0) + dot(mv1, mv1);
            }
        }
    }

    uint stride = 2 * p.rank + 4;
    device float* to = down + (job * p.n + t0) * stride;
    for (uint t = 0; t < count; t++) {
        from_input[t] = simd_sum(from_input[t]);
        from_message[t] = simd_sum(from_message[t]);
        if (norms) {
            squares[t] = simd_sum(squares[t]);
            message_squares[t] = simd_sum(message_squares[t]);
        }
        if (FEW && lane == 0) {
            partial[sid][4 * t] = from_input[t];
            partial[sid][4 * t + 1] = from_message[t];
            partial[sid][4 * t + 2] = squares[t];
            partial[sid][4 * t + 3] = message_squares[t];
        }
    }

    if (FEW) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid >= count) {
            return;
        }
        float fi = 0.0f;
        float fm = 0.0f;
        float sq = 0.0f;
        float msq = 0.0f;
        for (uint w = 0; w < nsg; w++) {
            fi += partial[w][4 * lid];
            fm += partial[w][4 * lid + 1];
            sq += partial[w][4 * lid + 2];
            msq += partial[w][4 * lid + 3];
        }
        device float* token = to + lid * stride;
        token[r] = fi;
        token[p.rank + r] = fm;
        if (norms) {
            token[2 * p.rank] = rsqrt(sq / float(p.dim) + p.eps);
            token[2 * p.rank + 1] = rsqrt(msq / float(p.dim) + p.eps);
        }
        return;
    }

    if (lane == 0) {
        for (uint t = 0; t < count; t++) {
            device float* token = to + t * stride;
            token[r] = from_input[t];
            token[p.rank + r] = from_message[t];
            if (norms) {
                token[2 * p.rank] = rsqrt(squares[t] / float(p.dim) + p.eps);
                token[2 * p.rank + 1] = rsqrt(message_squares[t] / float(p.dim) + p.eps);
            }
        }
    }
}
