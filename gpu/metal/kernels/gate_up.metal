// Second half of a message gate for up to GATES gates in one grid: reads
// the down-projected vectors and inverse RMS norms written by `gate_down`,
// projects back up through bf16 gain and forget matrices for up to
// GATE_TOKENS tokens at once, and updates x in place:
// x = x - tanh(forget) * x + tanh(gain) * message.

#include <metal_stdlib>
using namespace metal;

constant uint THREADS = 256;
constant uint GATE_TOKENS = 8;
constant uint GATE_RANK = 64;

struct Params {
    uint n;
    uint dim;
    uint rank;
    float eps;
};

static device float* pick_mut(
    device float* a, device float* b, device float* c, device float* d, uint j)
{
    switch (j) {
        case 0: return a;
        case 1: return b;
        case 2: return c;
        default: return d;
    }
}

static const device float* pick_f(
    const device float* a, const device float* b, const device float* c, const device float* d, uint j)
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

// tanh, which the GPU's own makes not a number of where its exponentials
// overflow: past 15 either way it is one to the last bit.
static float safe_tanh(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

kernel void gate_up(
    const device float* down [[buffer(0)]],
    device float* x0 [[buffer(1)]],
    device float* x1 [[buffer(2)]],
    device float* x2 [[buffer(3)]],
    device float* x3 [[buffer(4)]],
    const device float* m0 [[buffer(5)]],
    const device float* m1 [[buffer(6)]],
    const device float* m2 [[buffer(7)]],
    const device float* m3 [[buffer(8)]],
    const device uint4* g0 [[buffer(9)]],
    const device uint4* g1 [[buffer(10)]],
    const device uint4* g2 [[buffer(11)]],
    const device uint4* g3 [[buffer(12)]],
    const device uint4* f0 [[buffer(13)]],
    const device uint4* f1 [[buffer(14)]],
    const device uint4* f2 [[buffer(15)]],
    const device uint4* f3 [[buffer(16)]],
    constant Params& p [[buffer(17)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float4 downs[GATE_TOKENS * GATE_RANK / 4];
    uint blocks = (p.dim + THREADS - 1) / THREADS;
    uint job = wg.x / blocks;
    uint i = (wg.x % blocks) * THREADS + lid;
    uint t0 = wg.y * GATE_TOKENS;
    uint count = min(GATE_TOKENS, p.n - t0);
    uint per_row = p.rank / 8;
    uint quads = p.rank / 4;
    uint stride = 2 * p.rank + 4;

    // Load the normalized sum of the two down-projections for all tokens
    // in this tile into threadgroup memory.
    for (uint at = lid; at < count * quads; at += THREADS) {
        uint t = at / quads;
        uint q = (at % quads) * 4;
        const device float* from = down + (job * p.n + t0 + t) * stride;
        float norm_x = from[2 * p.rank];
        float norm_m = from[2 * p.rank + 1];
        float4 a = *(const device float4*)(from + q);
        float4 b = *(const device float4*)(from + p.rank + q);
        downs[at] = a * norm_x + b * norm_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i >= p.dim) {
        return;
    }

    const device uint4* gain = pick_u4(g0, g1, g2, g3, job);
    const device uint4* forget = pick_u4(f0, f1, f2, f3, job);
    float g[GATE_TOKENS] = {};
    float f[GATE_TOKENS] = {};
    for (uint c = 0; c < per_row; c++) {
        uint4 wg_w = gain[i * per_row + c];
        uint4 wf_w = forget[i * per_row + c];
        float4 wg0 = weights(wg_w.xy);
        float4 wg1 = weights(wg_w.zw);
        float4 wf0 = weights(wf_w.xy);
        float4 wf1 = weights(wf_w.zw);
        for (uint t = 0; t < count; t++) {
            float4 lo = downs[t * quads + 2 * c];
            float4 hi = downs[t * quads + 2 * c + 1];
            g[t] += dot(wg0, lo) + dot(wg1, hi);
            f[t] += dot(wf0, lo) + dot(wf1, hi);
        }
    }

    device float* x = pick_mut(x0, x1, x2, x3, job);
    const device float* message = pick_f(m0, m1, m2, m3, job);
    for (uint t = 0; t < count; t++) {
        uint at = (t0 + t) * p.dim + i;
        float xv = x[at];
        x[at] = xv - safe_tanh(f[t]) * xv + safe_tanh(g[t]) * message[at];
    }
}
