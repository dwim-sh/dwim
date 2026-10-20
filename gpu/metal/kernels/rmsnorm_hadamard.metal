// out = rotate(rmsnorm(x) * weight): one threadgroup of 256 per
// 1024-element block of every row of x, which sums the squares of the whole
// row, then normalizes, scales, and signs its block and transforms it as
// `hadamard.metal` does: each thread holds four consecutive elements, the
// first two rounds of butterflies are between them, the next five between
// lanes of a SIMD group by shuffles, and the last three between threads
// through shared memory. The row is read as vectors of four, a fixed number
// per thread with the index clamped rather than the load skipped, so that
// the loads go out together instead of one round trip after another.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    float eps;
};

constant uint BLOCK = 1024;
constant uint THREADS = 256;
// Vectors a thread reads of the row: rows of up to 8192 elements.
constant uint PER_THREAD = 8;

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

kernel void rmsnorm_hadamard(
    device float4* out [[buffer(0)]],
    const device float4* x [[buffer(1)]],
    const device float4* weight [[buffer(2)]],
    const device float4* signs [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    threadgroup float4 v[2 * THREADS];
    uint blocks = p.width / BLOCK;
    uint vectors = p.width / 4;
    uint row = (group / blocks) * vectors;
    uint block = (group % blocks) * (BLOCK / 4);
    // The thread's vector of the block, its weight and its signs, loaded
    // along with the row for the sum, so that nothing waits on the sum but
    // the arithmetic.
    float4 scaled = x[row + block + lid] * weight[block + lid] * signs[block + lid];
    float ss = 0.0f;
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = lid + k * THREADS;
        float4 e = x[row + min(i, vectors - 1)];
        ss += i < vectors ? dot(e, e) : 0.0f;
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
    out[row + block + lid] = butterfly(scaled * scale, lid, lane, v) * (1.0f / 32.0f);
}
