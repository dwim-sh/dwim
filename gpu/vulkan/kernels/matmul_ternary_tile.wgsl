// out[t][r] = w[r] · x[t] for ternary weights and a batch of many tokens,
// as a tiled matrix product. A workgroup takes 128 rows and 32 tokens. For
// each block of 128 columns it unpacks the rows' block once into workgroup
// memory, scaled and as half floats, which hold a scaled trit exactly,
// stages the tokens' activations beside it, and has each thread multiply
// eight rows into four tokens. The unpacking and the weight reads are so
// shared by 32 tokens, and the kernel is bound by multiply-adds. The
// weights are stored block-major, as `matmul_ternary.wgsl` describes. Fewer
// tokens than half a tile go through `matmul_ternary_batch.wgsl`.
//
// The next block's words and activations are fetched into registers
// before the multiplies of the current one, so that the memory latency
// hides behind them, and written to workgroup memory after.
//
// The activations sit by column, a vector of four tokens at a time, so
// that a thread's four tokens are one read; the weights sit by row with
// the pairs of a row swizzled by the row, so that the threads of a
// subgroup, which read different rows at the same pair, hit different
// banks.

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    // Workgroups per row of the dispatch grid, for matrices with more rows
    // than one dimension of the grid allows. The grid's second dimension
    // counts the token tiles of each such row.
    stride: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;

const THREADS: u32 = 128u;
// Rows and tokens of a tile, and the rows and tokens a thread multiplies.
const ROWS: u32 = 128u;
const TOKENS: u32 = 32u;
const MR: u32 = 8u;
const MT: u32 = 4u;
// Columns of a block, and the pairs of half floats they take.
const BLOCK: u32 = 128u;
const PAIRS: u32 = 64u;
// Words per block: 28 bytes.
const WORDS: u32 = 7u;
// Vectors of four activations a thread stages per block.
const FETCH: u32 = TOKENS * BLOCK / 4u / THREADS;

// The tile's weights for the block: a row's 128 columns as 64 pairs of
// half floats, at `wt_at`.
var<workgroup> wt: array<u32, ROWS * PAIRS>;
// The tile's activations for the block: column k's 32 tokens as eight
// vectors, from k * 8.
var<workgroup> xt: array<vec4<f32>, BLOCK * TOKENS / 4u>;

// Where pair `pair` of row `r` sits: the swizzle is even, so that a pair
// and the next stay adjacent.
fn wt_at(r: u32, pair: u32) -> u32 {
    return r * PAIRS + (pair ^ ((r & 15u) << 1u));
}

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
// 2, in the bytes' order: one more than the weight.
fn peel(s: ptr<function, Split>) -> vec4<f32> {
    let even = (*s).even * 3u;
    let odd = (*s).odd * 3u;
    (*s).even = even & 0x00ff00ffu;
    (*s).odd = odd & 0x00ff00ffu;
    return vec4(f32((even >> 8u) & 0xffu), f32((odd >> 8u) & 0xffu), f32(even >> 24u), f32(odd >> 24u));
}

// What a thread fetches of a block: the words of its row, and its share of
// the tokens' activations.
struct Fetch {
    words: array<u32, WORDS>,
    xs: array<vec4<f32>, FETCH>,
}

// Fetches block `b` of the tile from memory: a row per thread, zero past
// the matrix, and vectors of four columns of a token, zero past the batch.
fn fetch(b: u32, r0: u32, t0: u32, lid: u32) -> Fetch {
    var f: Fetch;
    let r = r0 + lid;
    if r < p.rows {
        let wbase = (b * p.rows + r) * WORDS;
        for (var j = 0u; j < WORDS; j++) {
            f.words[j] = w[wbase + j];
        }
    }
    for (var i = 0u; i < FETCH; i++) {
        let t = (lid + i * THREADS) % TOKENS;
        let k = ((lid + i * THREADS) / TOKENS) * 4u;
        if t0 + t < p.n {
            f.xs[i] = x[((t0 + t) * p.cols + b * BLOCK + k) / 4u];
        }
    }
    return f;
}

// Writes what a thread fetched to workgroup memory: the row's words
// unpacked and scaled, and the activations by column, the threads of a
// subgroup writing consecutive tokens.
fn stage(f: Fetch, lid: u32) {
    let row = lid;
    let scale = unpack2x16float(f.words[6]).y;
    // Words 0 to 3 hold elements 4j + i + 16n in byte 4j + i, and words 4
    // and 5 elements 80 + 4(j - 4) + i + 8n.
    for (var j = 0u; j < 4u; j++) {
        var s = split(f.words[j]);
        for (var n = 0u; n < 5u; n++) {
            let t = scale * (peel(&s) - 1.0);
            let pair = 2u * j + 8u * n;
            wt[wt_at(row, pair)] = pack2x16float(t.xy);
            wt[wt_at(row, pair + 1u)] = pack2x16float(t.zw);
        }
    }
    for (var j = 4u; j < 6u; j++) {
        var s = split(f.words[j]);
        for (var n = 0u; n < 5u; n++) {
            let t = scale * (peel(&s) - 1.0);
            let pair = 40u + 2u * (j - 4u) + 4u * n;
            wt[wt_at(row, pair)] = pack2x16float(t.xy);
            wt[wt_at(row, pair + 1u)] = pack2x16float(t.zw);
        }
    }
    // The low two bytes of the last word hold four trits each: byte 24 + i
    // holds elements 120 + i + 2n.
    var s = split(f.words[6] & 0xffffu);
    for (var n = 0u; n < 4u; n++) {
        let t = scale * (peel(&s) - 1.0);
        wt[wt_at(row, 60u + n)] = pack2x16float(t.xy);
    }
    for (var i = 0u; i < FETCH; i++) {
        let t = (lid + i * THREADS) % TOKENS;
        let k = ((lid + i * THREADS) / TOKENS) * 4u;
        for (var c = 0u; c < 4u; c++) {
            xt[(k + c) * 8u + t / 4u][t % 4u] = f.xs[i][c];
        }
    }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let token_tiles = (p.n + TOKENS - 1u) / TOKENS;
    let r0 = ((wg.y / token_tiles) * p.stride + wg.x) * ROWS;
    let t0 = (wg.y % token_tiles) * TOKENS;
    let blocks = p.cols / BLOCK;
    // The thread's rows are tr, tr + 16, ..., tr + 112, and its tokens the
    // vector tv: tokens 4tv to 4tv + 3.
    let tr = lid / 8u;
    let tv = lid % 8u;
    var acc: array<array<f32, MT>, MR>;
    var next = fetch(0u, r0, t0, lid);
    for (var b = 0u; b < blocks; b++) {
        workgroupBarrier();
        stage(next, lid);
        workgroupBarrier();
        if b + 1u < blocks {
            next = fetch(b + 1u, r0, t0, lid);
        }
        // The block's products are summed apart and added to the total
        // once, so that the rounding of a long sum does not swamp a small
        // total. Two pairs of columns at a time: a thread's eight rows are
        // eight reads of two adjacent pairs, and its four tokens at each of
        // the four columns are four reads of a vector.
        var sum: array<array<f32, MT>, MR>;
        for (var pair = 0u; pair < PAIRS; pair += 2u) {
            var wv: array<vec4<f32>, MR>;
            for (var m = 0u; m < MR; m++) {
                let at = wt_at(tr + 16u * m, pair);
                wv[m] = vec4(unpack2x16float(wt[at]), unpack2x16float(wt[at + 1u]));
            }
            var xv: array<vec4<f32>, 4>;
            for (var c = 0u; c < 4u; c++) {
                xv[c] = xt[(2u * pair + c) * 8u + tv];
            }
            // A column at a time across all the sums, so that consecutive
            // multiply-adds go to different sums and none waits on the
            // one before it.
            for (var c = 0u; c < 4u; c++) {
                for (var m = 0u; m < MR; m++) {
                    for (var q = 0u; q < MT; q++) {
                        sum[m][q] = fma(wv[m][c], xv[c][q], sum[m][q]);
                    }
                }
            }
        }
        for (var m = 0u; m < MR; m++) {
            for (var q = 0u; q < MT; q++) {
                acc[m][q] += sum[m][q];
            }
        }
    }
    for (var m = 0u; m < MR; m++) {
        let r = r0 + tr + 16u * m;
        for (var q = 0u; q < MT; q++) {
            let t = t0 + 4u * tv + q;
            if r < p.rows && t < p.n {
                out[t * p.rows + r] = acc[m][q];
            }
        }
    }
}
