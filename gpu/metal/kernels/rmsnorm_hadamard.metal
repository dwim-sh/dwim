// out = rotate(rmsnorm(x) * weight): one threadgroup per 1024-element
// block of every row of x, which sums the squares of the whole row, then
// normalizes, scales, and signs its block in shared memory and transforms
// it there as ten rounds of butterflies, as `hadamard.metal` does.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    float eps;
};

constant uint BLOCK = 1024;

kernel void rmsnorm_hadamard(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    const device float* signs [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    threadgroup float v[1024];
    uint blocks = p.width / BLOCK;
    uint row = (group / blocks) * p.width;
    uint block = (group % blocks) * BLOCK;
    float ss = 0.0f;
    for (uint i = lid; i < p.width; i += threads) {
        float e = x[row + i];
        ss += e * e;
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        partial[sid] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / float(p.width) + p.eps);
    for (uint i = lid; i < BLOCK; i += threads) {
        v[i] = x[row + block + i] * scale * weight[block + i] * signs[block + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < BLOCK; h <<= 1) {
        for (uint pair = lid; pair < BLOCK / 2; pair += threads) {
            uint i = (pair / h) * 2 * h + (pair % h);
            float a = v[i];
            float b = v[i + h];
            v[i] = a + b;
            v[i + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = lid; i < BLOCK; i += threads) {
        out[row + block + i] = v[i] * (1.0f / 32.0f);
    }
}
