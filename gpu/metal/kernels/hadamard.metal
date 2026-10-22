// Rotates each 1024-element block of every row of x by the normalized
// Walsh-Hadamard transform, multiplying by the signs before it (the forward
// rotation) or after it (the inverse): one threadgroup of 256 per block.
// Each thread holds four consecutive elements: the first two rounds of
// butterflies are between them, the next five between lanes of a SIMD group
// by shuffles, and the last three between threads through shared memory,
// each in one of two arrays in turn so that a round needs one barrier.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    uint inverse;
};

constant uint BLOCK = 1024;
constant uint THREADS = 256;

// The Walsh-Hadamard transform of the block whose elements 4lid to 4lid + 3
// are `e`, unnormalized.
static float4 butterfly(float4 e, uint lid, uint lane, threadgroup float4* v) {
    e = float4(e.x + e.y, e.x - e.y, e.z + e.w, e.z - e.w);
    e = float4(e.x + e.z, e.y + e.w, e.x - e.z, e.y - e.w);
    // Elements 4s apart are in the lanes s apart.
    for (ushort s = 1; s < 32; s <<= 1) {
        float4 o = simd_shuffle_xor(e, s);
        e = (lane & s) == 0 ? e + o : o - e;
    }
    uint side = 0;
    for (uint s = 32; s < THREADS; s <<= 1) {
        v[side + lid] = e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 o = v[side + (lid ^ s)];
        e = (lid & s) == 0 ? e + o : o - e;
        side = THREADS - side;
    }
    return e;
}

kernel void hadamard(
    device float4* x [[buffer(0)]],
    const device float4* signs [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float4 v[2 * THREADS];
    uint blocks = p.width / BLOCK;
    uint base = ((group / blocks) * p.width + (group % blocks) * BLOCK) / 4 + lid;
    uint sbase = (group % blocks) * BLOCK / 4 + lid;
    float4 e = x[base];
    if (p.inverse == 0) {
        e *= signs[sbase];
    }
    e = butterfly(e, lid, lane, v) * (1.0f / 32.0f);
    if (p.inverse != 0) {
        e *= signs[sbase];
    }
    x[base] = e;
}
