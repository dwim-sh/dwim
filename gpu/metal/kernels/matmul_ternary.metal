// out[t][r] = w[r] · x[t] for ternary weights: each SIMD group takes eight
// rows, in groups of eight lanes each taking a block of 128 weights in
// turn. A group loads a block's worth of activations once and multiplies it
// into that block of each of its rows in turn, so the activations are read
// from memory once per eight rows rather than once per row. A block's trits
// are spread over its bytes so that the ones in the same position of four
// consecutive bytes belong to four consecutive elements: each lane of a
// group unpacks one word of the block, four bytes at a time, and
// multiplies each round of four trits into the four activations they
// belong to. The block layout is that of `ternary.rs`. Tokens go one after
// another; `matmul_ternary_batch.metal` takes several at once.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint n;
};

// Rows per SIMD group.
constant uint ROWS = 8;

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

// The dot product of word j of a block with the five vectors of
// activations its trits belong to, whose sum is xsum, times the block's
// scale.
static float word_dot(const device uint* block, uint j, thread float4* xs, float xsum) {
    float sum = -xsum;
    if (j < 6) {
        // Words 0 to 3 hold elements 4j + i + 16n in byte 4j + i, and words
        // 4 and 5 elements 80 + 4(j - 4) + i + 8n.
        Split s = split(block[j]);
        for (uint n = 0; n < 5; n++) {
            sum += dot(peel(s), xs[n]);
        }
    } else if (j == 6) {
        // The low two bytes of the last word: byte 24 + i holds elements
        // 120 + i + 2n, so each round of the two gives half a vector.
        Split s = split(block[6] & 0xffffu);
        for (uint n = 0; n < 2; n++) {
            float4 a = peel(s);
            float4 c = peel(s);
            sum += dot(float4(a.x, a.y, c.x, c.y), xs[n]);
        }
    }
    return float(as_type<half>(ushort(block[6] >> 16))) * sum;
}

kernel void matmul_ternary(
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
    for (uint t = 0; t < p.n; t++) {
        float acc[ROWS] = {};
        for (uint b = lane / 8; b < blocks; b += groups) {
            // The activations of the word's elements: five vectors, at
            // stride four for the first four words, two for the next two,
            // and one for the last.
            const device float4* xb = x + (t * p.cols + b * 128) / 4;
            float4 xs[5];
            float xsum = 0.0f;
            if (j < 4) {
                for (uint n = 0; n < 5; n++) {
                    xs[n] = xb[j + 4 * n];
                    xsum += dot(xs[n], float4(1.0f));
                }
            } else if (j < 6) {
                for (uint n = 0; n < 5; n++) {
                    xs[n] = xb[20 + (j - 4) + 2 * n];
                    xsum += dot(xs[n], float4(1.0f));
                }
            } else if (j == 6) {
                xs[0] = xb[30];
                xs[1] = xb[31];
                xsum = dot(xs[0] + xs[1], float4(1.0f));
            }
            for (uint r = 0; r < ROWS; r++) {
                if (r0 + r < p.rows) {
                    acc[r] += word_dot(w + ((r0 + r) * blocks + b) * 7, j, xs, xsum);
                }
            }
        }
        for (uint r = 0; r < ROWS; r++) {
            float total = simd_sum(acc[r]);
            if (lane == 0 && r0 + r < p.rows) {
                out[t * p.rows + r0 + r] = total;
            }
        }
    }
}
