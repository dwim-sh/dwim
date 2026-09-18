// The gated delta rule of linear attention: one threadgroup per value head,
// stepping through the tokens in order. The head's state is a matrix of
// keys by values; each thread keeps a column of half of it in registers, so
// what the state holds for a key is summed over the two halves through
// shared memory, as is the output for the query.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_k_heads;
    uint n_v_heads;
    uint head_dim;
};

// The head dimension the kernel is written for: 256 threads, each holding
// 64 of the 128 by 128 state.
constant uint DIM = 128;
constant uint HALF = 64;

kernel void delta_net(
    device float* out [[buffer(0)]],
    const device float4* q [[buffer(1)]],
    const device float4* k [[buffer(2)]],
    const device float* v [[buffer(3)]],
    const device float* gates [[buffer(4)]],
    const device float* decay [[buffer(5)]],
    device float* state [[buffer(6)]],
    constant Params& p [[buffer(7)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float kv[256];
    threadgroup float oq[256];
    uint j = lid % DIM;
    uint half_ = lid / DIM;
    uint kh = h / (p.n_v_heads / p.n_k_heads);
    uint sbase = h * DIM * DIM;
    float4 s[16];
    for (uint i = 0; i < 16; i++) {
        uint row = half_ * HALF + 4 * i;
        s[i] = float4(state[sbase + row * DIM + j], state[sbase + (row + 1) * DIM + j],
                      state[sbase + (row + 2) * DIM + j], state[sbase + (row + 3) * DIM + j]);
    }
    float scale = rsqrt(float(DIM));
    float a = decay[h];
    float dt_bias = decay[p.n_v_heads + h];
    for (uint t = 0; t < p.n; t++) {
        float alpha = gates[t * 2 * p.n_v_heads + h];
        float beta_raw = gates[t * 2 * p.n_v_heads + p.n_v_heads + h];
        float z = alpha + dt_bias;
        float softplus = z > 20.0f ? z : log(1.0f + exp(z));
        float g = exp(a * softplus);
        float beta = 1.0f / (1.0f + exp(-beta_raw));
        // The state decays, and what it holds for the key is read out.
        uint kbase = ((t * p.n_k_heads + kh) * DIM + half_ * HALF) / 4;
        float mem = 0.0f;
        for (uint i = 0; i < 16; i++) {
            s[i] *= g;
            mem += dot(s[i], k[kbase + i]);
        }
        kv[lid] = mem;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float delta = (v[(t * p.n_v_heads + h) * DIM + j] - kv[j] - kv[DIM + j]) * beta;
        // The value takes its place, and the query reads the state out.
        uint qbase = ((t * p.n_k_heads + kh) * DIM + half_ * HALF) / 4;
        float o = 0.0f;
        for (uint i = 0; i < 16; i++) {
            s[i] += k[kbase + i] * delta;
            o += dot(s[i], q[qbase + i]);
        }
        oq[lid] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (half_ == 0) {
            out[(t * p.n_v_heads + h) * DIM + j] = (oq[j] + oq[DIM + j]) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 16; i++) {
        uint row = half_ * HALF + 4 * i;
        state[sbase + row * DIM + j] = s[i].x;
        state[sbase + (row + 1) * DIM + j] = s[i].y;
        state[sbase + (row + 2) * DIM + j] = s[i].z;
        state[sbase + (row + 3) * DIM + j] = s[i].w;
    }
}
