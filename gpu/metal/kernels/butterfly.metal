// The Walsh-Hadamard transform of a 1024-element block by a threadgroup of
// 256, for `hadamard.metal` and `rmsnorm_hadamard.metal`, which the device
// puts ahead of their source. Each thread holds four consecutive elements:
// the first two rounds of butterflies are between them, the next five
// between lanes of a SIMD group by shuffles, and the last three between
// threads through shared memory, each in one of two arrays in turn so that
// a round needs one barrier.

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK = 1024;
constant uint THREADS = 256;

// The transform of the block whose elements 4lid to 4lid + 3 are `e`,
// unnormalized, through `v`, which holds two arrays of a vector per thread.
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
