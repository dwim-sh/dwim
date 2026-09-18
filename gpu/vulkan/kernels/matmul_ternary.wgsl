// out[t][r] = w[r] · x[t] for ternary weights: one workgroup per eight rows,
// in eight groups of eight threads, each group taking every eighth block of
// 128 weights. A group loads a block's worth of activations once and
// multiplies it into that block of each of its rows in turn, so the
// activations are read from memory once per eight rows rather than once per
// row. A block's trits are spread over its bytes so that the ones in the
// same position of four consecutive bytes belong to four consecutive
// elements: each thread of a group unpacks one word of the block, four
// bytes at a time, and multiplies each round of four trits into the four
// activations they belong to. The block layout is that of `ternary.rs`,
// and the blocks are stored block-major, a block of every row before the
// next block, so that a group's eight rows read adjacent blocks. Tokens go
// one after another; `matmul_ternary_batch.wgsl` takes several at once.

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    // Workgroups per row of the dispatch grid, for matrices with more rows
    // than one dimension of the grid allows.
    stride: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;

// Rows per workgroup.
const ROWS: u32 = 8u;

// Words per block: 28 bytes.
const WORDS: u32 = 7u;

var<workgroup> partial: array<f32, 32>;

// The four bytes of a word as two pairs in 16-bit halves, so that tripling
// them does not carry between bytes.
struct Split {
    even: u32,
    odd: u32,
}

fn split(word: u32) -> Split {
    return Split(word & 0x00ff00ffu, (word >> 8u) & 0x00ff00ffu);
}

// Peels the most significant trit off each of the four bytes, as 0, 1, or
// 2, in the bytes' order: one more than the weight, which is taken off the
// sum at the end.
fn peel(s: ptr<function, Split>) -> vec4<f32> {
    let even = (*s).even * 3u;
    let odd = (*s).odd * 3u;
    (*s).even = even & 0x00ff00ffu;
    (*s).odd = odd & 0x00ff00ffu;
    return vec4(f32((even >> 8u) & 0xffu), f32((odd >> 8u) & 0xffu), f32(even >> 24u), f32(odd >> 24u));
}

// sum + a · b, as four fused multiply-adds.
fn mad(sum: f32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, sum))));
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let r0 = (wg.y * p.stride + wg.x) * ROWS;
    let blocks = p.cols / 128u;
    let group = lid / 8u;
    let j = lid % 8u;
    for (var t = 0u; t < p.n; t++) {
        var acc: array<f32, ROWS>;
        for (var b = group; b < blocks; b += 8u) {
            // The activations of the word's elements: five vectors, at
            // stride four for the first four words, two for the next two,
            // and one for the last.
            let xbase = (t * p.cols + b * 128u) / 4u;
            var xs: array<vec4<f32>, 5>;
            var xsum = 0.0;
            if j < 4u {
                for (var n = 0u; n < 5u; n++) {
                    xs[n] = x[xbase + j + 4u * n];
                    xsum += dot(xs[n], vec4(1.0));
                }
            } else if j < 6u {
                for (var n = 0u; n < 5u; n++) {
                    xs[n] = x[xbase + 20u + (j - 4u) + 2u * n];
                    xsum += dot(xs[n], vec4(1.0));
                }
            } else if j == 6u {
                xs[0] = x[xbase + 30u];
                xs[1] = x[xbase + 31u];
                xsum = dot(xs[0] + xs[1], vec4(1.0));
            }
            // Word j of the block of each row, multiplied into the
            // activations, less the trits' offset, times the block's
            // scale.
            for (var r = 0u; r < ROWS; r++) {
                if r0 + r >= p.rows {
                    continue;
                }
                let wbase = (b * p.rows + r0 + r) * WORDS;
                var sum = -xsum;
                if j < 6u {
                    // Words 0 to 3 hold elements 4j + i + 16n in byte
                    // 4j + i, and words 4 and 5 elements 80 + 4(j - 4) + i
                    // + 8n.
                    var s = split(w[wbase + j]);
                    for (var n = 0u; n < 5u; n++) {
                        sum = mad(sum, peel(&s), xs[n]);
                    }
                } else if j == 6u {
                    // The low two bytes of the last word: byte 24 + i holds
                    // elements 120 + i + 2n, so each round of the two gives
                    // half a vector.
                    var s = split(w[wbase + 6u] & 0xffffu);
                    for (var n = 0u; n < 2u; n++) {
                        let a = peel(&s);
                        let c = peel(&s);
                        sum = mad(sum, vec4(a.x, a.y, c.x, c.y), xs[n]);
                    }
                }
                acc[r] += unpack2x16float(w[wbase + 6u]).y * sum;
            }
        }
        for (var r = 0u; r < ROWS; r++) {
            let s = subgroupAdd(acc[r]);
            if sinv == 0u {
                partial[sid * ROWS + r] = s;
            }
        }
        workgroupBarrier();
        if lid < ROWS && r0 + lid < p.rows {
            var total = 0.0;
            for (var i = 0u; i < nsg; i++) {
                total += partial[i * ROWS + lid];
            }
            out[t * p.rows + r0 + lid] = total;
        }
        workgroupBarrier();
    }
}
