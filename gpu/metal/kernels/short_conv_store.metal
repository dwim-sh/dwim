// Fused short causal convolution, head clip or RMS norm, ring-buffer
// wrap, and 8-bit cache quantization: one threadgroup per (token, head),
// one thread per channel, each SIMD group quantizing one 32-element cache
// block. A token alone moves its convolution state on in place; for a
// batch, `short_conv_state.metal` updates it afterwards.

#include <metal_stdlib>
using namespace metal;

constant uint KEPT_CLIPPED = 0;
constant uint CACHE_BLOCK = 32;

struct Params {
    uint n;
    uint channels;
    uint taps;
    uint heads;
    uint dim;
    uint width;
    uint kept;
    uint pos;
    uint rows;
    float value;
};

static float input(const device float* x, device float* state, uint channels, uint taps, uint t, uint back, uint c) {
    if (back > t) {
        return state[(taps - 1 + t - back) * channels + c];
    }
    return x[(t - back) * channels + c];
}

kernel void short_conv_store(
    device char* quants [[buffer(0)]],
    device half* scales [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* state [[buffer(3)]],
    const device float* weight [[buffer(4)]],
    const device float* dynamic [[buffer(5)]],
    const device float* by [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    uint per_token = p.channels / p.dim;
    uint t = wg / per_token;
    uint c = (wg % per_token) * p.dim + lid;

    uint head = p.heads != 0 ? c / (p.channels / p.heads) : 0;
    float acc = x[t * p.channels + c];
    for (uint lag = 0; lag < p.taps; lag++) {
        float w = weight[c * p.taps + (p.taps - 1 - lag)];
        if (p.heads != 0) {
            w += dynamic[(t * p.heads + head) * p.taps + lag];
        }
        acc += w * input(x, state, p.channels, p.taps, t, lag, c);
    }
    if (p.n == 1 && p.taps > 1) {
        for (uint s = 0; s + 2 < p.taps; s++) {
            state[s * p.channels + c] = state[(s + 1) * p.channels + c];
        }
        state[(p.taps - 2) * p.channels + c] = x[c];
    }
    float v = acc * rsqrt(float((p.heads != 0 ? 2 : 1) * p.taps + 1));

    if (p.kept == KEPT_CLIPPED) {
        v *= by[c % p.width];
    }
    float sq = simd_sum(v * v);
    if (lane == 0) {
        partial[sid] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; i++) {
        total += partial[i];
    }
    if (p.kept == KEPT_CLIPPED) {
        v *= min(p.value / (sqrt(total) + 1e-6f), 1.0f);
    } else {
        v = v * rsqrt(total / float(p.dim) + p.value) * by[c % p.dim];
    }

    // Quantize this SIMD group's 32-element block into the cache ring.
    float a = fabs(v);
    float amax = simd_max(a);
    half scale = half(min(amax * (1.0f / 127.0f), 65504.0f));
    float d = float(scale);
    float q = 0.0f;
    if (d > 0.0f) {
        q = min(rint(a / d), 128.0f);
        if (a >= (q + 0.5f) * d) {
            q += 1.0f;
        } else if (q > 0.0f && a < (q - 0.5f) * d) {
            q -= 1.0f;
        }
        q = min(q, 127.0f);
    }
    uint at = ((p.pos + t) % p.rows) * p.channels + c;
    quants[at] = char(v < 0.0f ? -q : q);
    if (lane == 0) {
        scales[at / CACHE_BLOCK] = scale;
    }
}
