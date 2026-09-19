// out[t][r] = w[r] · x[t] for ternary weights: one workgroup per eight rows,
// in eight groups of eight threads, each group taking every eighth block of
// 128 weights. A group loads a block's worth of activations once and
// multiplies it into that block of each of its rows in turn, so the
// activations are read from memory once per eight rows rather than once per
// row, and it loads the words of all eight rows before multiplying any, so
// that the loads are in flight together. A block's trits are spread over
// its bytes so that the ones in the same position of four consecutive
// bytes belong to four consecutive elements: each thread of a group takes
// one word of the block and looks each of its bytes up in a table of the
// five trits a byte holds, as half floats, so that a trit costs one
// multiply-add and no arithmetic to unpack. The table is 4 KB, built once
// per workgroup. The block layout is that of `ternary.rs`, and the blocks
// are stored block-major, a block of every row before the next block, so
// that a group's eight rows read adjacent blocks. Tokens go one after
// another; `matmul_ternary_batch.wgsl` takes several at once.

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    // Workgroups per row of the dispatch grid, for matrices with more rows
    // than one dimension of the grid allows.
    stride: u32,
    // Column splits, which only the tile kernel takes.
    splits: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;

// Threads and rows per workgroup.
const THREADS: u32 = 64u;
const ROWS: u32 = 8u;

// Words per block: 28 bytes.
const WORDS: u32 = 7u;

// Each subgroup's sums of the rows, for the workgroup to add up.
var<workgroup> partial: array<f32, 32>;

// The trits of each byte value, most significant first, as half floats
// -1, 0 and 1 in pairs: the first two in x, the next two in y, the fifth
// in the low half of z. A byte of a block holds five trits as a base-3
// number scaled to fill the byte, so that tripling it carries the leading
// trit into the ninth bit; the last two bytes of a block hold four, laid
// out the same way, so the same table serves them.
var<workgroup> table: array<vec4<u32>, 256>;

fn build(lid: u32) {
    for (var v = lid; v < 256u; v += THREADS) {
        var q = v;
        var t: array<f32, 5>;
        for (var n = 0u; n < 5u; n++) {
            t[n] = f32((q * 3u) >> 8u) - 1.0;
            q = (q * 3u) & 0xffu;
        }
        table[v] = vec4(pack2x16float(vec2(t[0], t[1])), pack2x16float(vec2(t[2], t[3])), pack2x16float(vec2(t[4], 0.0)), 0u);
    }
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
    let groups = THREADS / 8u;
    let group = lid / 8u;
    let j = lid % 8u;
    build(lid);
    workgroupBarrier();
    for (var t = 0u; t < p.n; t++) {
        var acc: array<f32, ROWS>;
        for (var b = group; b < blocks; b += groups) {
            // The activations of the word's elements: five vectors, at
            // stride four for the first four words, two for the next two,
            // and one for the last.
            let xbase = (t * p.cols + b * 128u) / 4u;
            var xs: array<vec4<f32>, 5>;
            if j < 4u {
                for (var n = 0u; n < 5u; n++) {
                    xs[n] = x[xbase + j + 4u * n];
                }
            } else if j < 6u {
                for (var n = 0u; n < 5u; n++) {
                    xs[n] = x[xbase + 20u + (j - 4u) + 2u * n];
                }
            } else if j == 6u {
                xs[0] = x[xbase + 30u];
                xs[1] = x[xbase + 31u];
            }
            // The words of all eight rows first, so that their loads are
            // in flight together. A row past the matrix reads the last row
            // instead, in bounds, and its sum goes nowhere. The scale is in
            // the last word, which the group's seventh lane has.
            var words: array<u32, ROWS>;
            var scales: array<f32, ROWS>;
            for (var r = 0u; r < ROWS; r++) {
                let wbase = (b * p.rows + min(r0 + r, p.rows - 1u)) * WORDS;
                words[r] = w[wbase + min(j, 6u)];
            }
            for (var r = 0u; r < ROWS; r++) {
                scales[r] = unpack2x16float(subgroupShuffle(words[r], (sinv & ~7u) + 6u)).y;
            }
            for (var r = 0u; r < ROWS; r++) {
                let word = words[r];
                var sum = 0.0;
                if j < 6u {
                    // Words 0 to 3 hold elements 4j + i + 16n in byte
                    // 4j + i, and words 4 and 5 elements 80 + 4(j - 4) + i
                    // + 8n: byte i's trit n goes with xs[n][i].
                    for (var i = 0u; i < 4u; i++) {
                        let e = table[(word >> (8u * i)) & 0xffu];
                        let t01 = unpack2x16float(e.x);
                        let t23 = unpack2x16float(e.y);
                        let t4 = unpack2x16float(e.z).x;
                        sum = fma(t01.x, xs[0][i], sum);
                        sum = fma(t01.y, xs[1][i], sum);
                        sum = fma(t23.x, xs[2][i], sum);
                        sum = fma(t23.y, xs[3][i], sum);
                        sum = fma(t4, xs[4][i], sum);
                    }
                } else if j == 6u {
                    // The low two bytes of the last word hold four trits
                    // each: byte 24 + i holds elements 120 + i + 2n.
                    for (var i = 0u; i < 2u; i++) {
                        let e = table[(word >> (8u * i)) & 0xffu];
                        let t01 = unpack2x16float(e.x);
                        let t23 = unpack2x16float(e.y);
                        sum = fma(t01.x, xs[0][i], sum);
                        sum = fma(t01.y, xs[0][2u + i], sum);
                        sum = fma(t23.x, xs[1][i], sum);
                        sum = fma(t23.y, xs[1][2u + i], sum);
                    }
                }
                acc[r] += scales[r] * sum;
            }
        }
        // The groups covered every block between them: each subgroup sums
        // its lanes, and the workgroup the subgroups.
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
