// out[i][r] = scale[e] * (w[e][r] · x[i]) for pair i of a token and one of
// its experts, e: a matrix multiplication by a different matrix for each
// pair, out of the experts' matrices one after another in w. Each SIMD
// group takes one row of one pair, its lanes a block of the row's weights
// each in turn. The input is a row per token, shared by its experts, or a
// row per pair.
//
// The weights are NVFP4 or Q8_0, as BITS says, laid out as `quant.rs`
// says. An NVFP4 block is 16 weights in two 4-byte words, byte j holding
// element j in its low nibble and element j + 8 in its high one, each one
// of sixteen values, and its scale a byte: a float of four bits of
// exponent over three of mantissa.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    // Experts per token, and whether the input is a row per token.
    uint k;
    uint shared;
};

#if BITS == 4

constant uint BLOCK = 16;

// What a weight's four bits stand for, doubled: the scale is halved.
constant float VALUES[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// The low nibbles of a word's four bytes as weights, in memory order.
static float4 low(uint word) {
    return float4(VALUES[word & 0xfu], VALUES[(word >> 8) & 0xfu], VALUES[(word >> 16) & 0xfu], VALUES[(word >> 24) & 0xfu]);
}

// The high nibbles, likewise.
static float4 high(uint word) {
    return float4(VALUES[(word >> 4) & 0xfu], VALUES[(word >> 12) & 0xfu], VALUES[(word >> 20) & 0xfu], VALUES[word >> 28]);
}

// A block's scale from its bits, halved. Zero and the bits of a NaN scale
// the block to nothing.
static float scale_of(uchar bits) {
    if (bits == 0 || bits == 0x7f) {
        return 0.0f;
    }
    int e = int((bits >> 3) & 0xf);
    float m = float(bits & 7);
    return e == 0 ? m * (0.5f / 512.0f) : ldexp(0.5f + m / 16.0f, e - 7);
}

// The dot product of block b of a row with the same block of the input.
static float block_dot(const device uint2* w, const device uchar* scales, const device float4* x, uint row, uint blocks, uint b) {
    uint2 v = w[row * blocks + b];
    const device float4* xb = x + b * 4;
    float sum = dot(low(v.x), xb[0]) + dot(low(v.y), xb[1]) + dot(high(v.x), xb[2]) + dot(high(v.y), xb[3]);
    return scale_of(scales[row * blocks + b]) * sum;
}

#define WORDS uint2
#define SCALE uchar

#else

constant uint BLOCK = 32;

// The four signed bytes of a word, in memory order.
static float4 unpack(uint word) {
    return float4(as_type<char4>(word));
}

static float block_dot(const device uint4* w, const device half* scales, const device float4* x, uint row, uint blocks, uint b) {
    uint4 lo = w[(row * blocks + b) * 2];
    uint4 hi = w[(row * blocks + b) * 2 + 1];
    const device float4* xb = x + b * 8;
    float sum = dot(unpack(lo.x), xb[0]) + dot(unpack(lo.y), xb[1]) + dot(unpack(lo.z), xb[2]) + dot(unpack(lo.w), xb[3])
        + dot(unpack(hi.x), xb[4]) + dot(unpack(hi.y), xb[5]) + dot(unpack(hi.z), xb[6]) + dot(unpack(hi.w), xb[7]);
    return float(scales[row * blocks + b]) * sum;
}

#define WORDS uint4
#define SCALE half

#endif

kernel void expert_matmul(
    device float* out [[buffer(0)]],
    const device WORDS* w [[buffer(1)]],
    const device SCALE* scales [[buffer(2)]],
    const device float4* x [[buffer(3)]],
    const device float* ids [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    constant Params& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    uint r = group.x * nsg + sid;
    uint pair = group.y;
    if (r >= p.rows) {
        return;
    }
    uint expert = uint(ids[pair]);
    uint row = expert * p.rows + r;
    const device float4* xr = x + (p.shared != 0 ? pair / p.k : pair) * (p.cols / 4);
    uint blocks = p.cols / BLOCK;
    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += width) {
        acc += block_dot(w, scales, xr, row, blocks, b);
    }
    float total = simd_sum(acc);
    if (lane == 0) {
        out[pair * p.rows + r] = scale[expert] * total;
    }
}
