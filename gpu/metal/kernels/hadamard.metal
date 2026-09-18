// Rotates each 1024-element block of every row of x by the normalized
// Walsh-Hadamard transform, multiplying by the signs before it (the forward
// rotation) or after it (the inverse): one threadgroup per block, which it
// transforms in shared memory as ten rounds of butterflies.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    uint inverse;
};

constant uint BLOCK = 1024;

kernel void hadamard(
    device float* x [[buffer(0)]],
    const device float* signs [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    threadgroup float v[1024];
    uint blocks = p.width / BLOCK;
    uint base = (group / blocks) * p.width + (group % blocks) * BLOCK;
    uint sbase = (group % blocks) * BLOCK;
    for (uint i = lid; i < BLOCK; i += threads) {
        float e = x[base + i];
        if (p.inverse == 0) {
            e *= signs[sbase + i];
        }
        v[i] = e;
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
        float e = v[i] * (1.0f / 32.0f);
        if (p.inverse != 0) {
            e *= signs[sbase + i];
        }
        x[base + i] = e;
    }
}
