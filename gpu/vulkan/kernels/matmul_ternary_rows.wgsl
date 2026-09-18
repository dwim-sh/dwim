// out[t][r] = w[r] · x[t] for ternary weights: one lane per row, the
// subgroups of a workgroup each taking a share of the columns, so that a
// subgroup's lanes read the same activations at the same time, and a lane
// walks its row's blocks one after another. Faster than
// `matmul_ternary.wgsl` for matrices of few rows and many columns, where
// each lane's stretch of its row is long, and slower for tall ones. The
// block layout is that of `ternary.rs`, and the blocks are stored
// block-major, as `matmul_ternary.wgsl` describes, so that the lanes of a
// subgroup read adjacent blocks.

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

const WORDS: u32 = 7u;

var<workgroup> partial: array<f32, 256>;

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

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
    @builtin(subgroup_size) ssize: u32,
) {
    let r = (wg.y * p.stride + wg.x) * ssize + sinv;
    let blocks = p.cols / 128u;
    // This subgroup's share of the blocks.
    let first = sid * blocks / nsg;
    let last = (sid + 1u) * blocks / nsg;
    for (var t = 0u; t < p.n; t++) {
        var acc = 0.0;
        for (var b = first; b < last; b++) {
            let xbase = (t * p.cols + b * 128u) / 4u;
            var xsum = 0.0;
            var sum = 0.0;
            if r < p.rows {
                let wbase = (b * p.rows + r) * WORDS;
                var words: array<u32, 7>;
                for (var i = 0u; i < WORDS; i++) {
                    words[i] = w[wbase + i];
                }
                // Words 0 to 3 hold elements 4k + i + 16n in byte 4k + i,
                // words 4 and 5 elements 80 + 4(k - 4) + i + 8n, and the low
                // two bytes of word 6 elements 120 + i + 2n.
                for (var k = 0u; k < 4u; k++) {
                    var s = split(words[k]);
                    for (var n = 0u; n < 5u; n++) {
                        let xv = x[xbase + k + 4u * n];
                        xsum += dot(xv, vec4(1.0));
                        sum = mad(sum, peel(&s), xv);
                    }
                }
                for (var k = 4u; k < 6u; k++) {
                    var s = split(words[k]);
                    for (var n = 0u; n < 5u; n++) {
                        let xv = x[xbase + 20u + (k - 4u) + 2u * n];
                        xsum += dot(xv, vec4(1.0));
                        sum = mad(sum, peel(&s), xv);
                    }
                }
                var s = split(words[6] & 0xffffu);
                for (var n = 0u; n < 2u; n++) {
                    let a = peel(&s);
                    let c = peel(&s);
                    let xv = x[xbase + 30u + n];
                    xsum += dot(xv, vec4(1.0));
                    sum = mad(sum, vec4(a.x, a.y, c.x, c.y), xv);
                }
                acc += unpack2x16float(words[6]).y * (sum - xsum);
            }
        }
        partial[sid * ssize + sinv] = acc;
        workgroupBarrier();
        if sid == 0u && r < p.rows {
            var total = 0.0;
            for (var i = 0u; i < nsg; i++) {
                total += partial[i * ssize + sinv];
            }
            out[t * p.rows + r] = total;
        }
        workgroupBarrier();
    }
}
