// out[t][r] = w[r] · x[t] for ternary weights and a batch of tokens: as
// `matmul_ternary.metal`, but taking the tokens four at a time, so that a
// block is unpacked once per four tokens rather than once per token. A
// last, shorter group of tokens has zero activations for the ones it
// lacks, and costs the same.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
};

// Rows per SIMD group, and tokens taken at once.
constant uint ROWS = 8;
constant uint TOKENS = 4;

// The four bytes of a word as two pairs in 16-bit halves, so that tripling
// them does not carry between bytes.
struct Split {
    uint even;
    uint odd;
};

static Split split(uint word) {
    return Split { word & 0x00ff00ffu, (word >> 8) & 0x00ff00ffu };
}

// Peels the most significant trit off each of the four bytes, as 0, 1, or
// 2, in the bytes' order: one more than the weight, which is taken off the
// sum at the end.
static float4 peel(thread Split& s) {
    uint even = s.even * 3;
    uint odd = s.odd * 3;
    s.even = even & 0x00ff00ffu;
    s.odd = odd & 0x00ff00ffu;
    return float4(float((even >> 8) & 0xff), float((odd >> 8) & 0xff), float(even >> 24), float(odd >> 24));
}

kernel void matmul_ternary_batch(
    device float* out [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device float4* x [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
    uint r0 = (group * nsg + sid) * ROWS;
    if (r0 >= p.rows) {
        return;
    }
    uint blocks = p.cols / 128;
    uint groups = width / 8;
    uint j = lane % 8;
    for (uint t0 = 0; t0 < p.n; t0 += TOKENS) {
        float acc[ROWS][TOKENS] = {};
        for (uint b = lane / 8; b < blocks; b += groups) {
            // The activations of the word's elements for each token: five
            // vectors, at stride four for the first four words, two for the
            // next two, and one for the last.
            float4 xs[TOKENS][5] = {};
            float xsum[TOKENS] = {};
            for (uint k = 0; k < TOKENS; k++) {
                if (t0 + k >= p.n) {
                    continue;
                }
                const device float4* xb = x + ((t0 + k) * p.cols + b * 128) / 4;
                if (j < 4) {
                    for (uint n = 0; n < 5; n++) {
                        xs[k][n] = xb[j + 4 * n];
                        xsum[k] += dot(xs[k][n], float4(1.0f));
                    }
                } else if (j < 6) {
                    for (uint n = 0; n < 5; n++) {
                        xs[k][n] = xb[20 + (j - 4) + 2 * n];
                        xsum[k] += dot(xs[k][n], float4(1.0f));
                    }
                } else if (j == 6) {
                    xs[k][0] = xb[30];
                    xs[k][1] = xb[31];
                    xsum[k] = dot(xs[k][0] + xs[k][1], float4(1.0f));
                }
            }
            // Word j of the block of each row, multiplied into the four
            // tokens' activations, less the trits' offset, times the
            // block's scale.
            for (uint r = 0; r < ROWS; r++) {
                if (r0 + r >= p.rows) {
                    continue;
                }
                const device uint* block = w + ((r0 + r) * blocks + b) * 7;
                float sum[TOKENS];
                for (uint k = 0; k < TOKENS; k++) {
                    sum[k] = -xsum[k];
                }
                if (j < 6) {
                    Split s = split(block[j]);
                    for (uint n = 0; n < 5; n++) {
                        float4 trits = peel(s);
                        for (uint k = 0; k < TOKENS; k++) {
                            sum[k] += dot(trits, xs[k][n]);
                        }
                    }
                } else if (j == 6) {
                    Split s = split(block[6] & 0xffffu);
                    for (uint n = 0; n < 2; n++) {
                        float4 a = peel(s);
                        float4 c = peel(s);
                        float4 trits = float4(a.x, a.y, c.x, c.y);
                        for (uint k = 0; k < TOKENS; k++) {
                            sum[k] += dot(trits, xs[k][n]);
                        }
                    }
                }
                float d = float(as_type<half>(ushort(block[6] >> 16)));
                for (uint k = 0; k < TOKENS; k++) {
                    acc[r][k] += d * sum[k];
                }
            }
        }
        for (uint k = 0; k < TOKENS; k++) {
            for (uint r = 0; r < ROWS; r++) {
                float total = simd_sum(acc[r][k]);
                if (lane == 0 && r0 + r < p.rows && t0 + k < p.n) {
                    out[(t0 + k) * p.rows + r0 + r] = total;
                }
            }
        }
    }
}
