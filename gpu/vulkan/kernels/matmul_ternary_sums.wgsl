// out[r] = w[r] · x for ternary weights and a single token, by looking each
// byte of a block up in a table of what its trits' elements of the token
// sum to. A byte holds five trits of five fixed elements of its block, so
// for each of a block's 26 byte positions there are 256 sums the byte can
// stand for, and every row shares them: a workgroup builds the 26 tables of
// a block in workgroup memory, then each of its rows costs one table read
// and one add per byte, with no unpacking and no multiplies. A workgroup
// takes 1024 rows, eight per thread, over a range of blocks; the ranges of
// a matrix with too few rows to fill the GPU are split among workgroups,
// whose partial sums `matmul_ternary_tile_reduce.wgsl` adds up. The words of a row's
// block are loaded before the tables are built, so the loads are in
// flight through the building. The block layout is that of `ternary.rs`,
// and the blocks are stored block-major, a block of every row before the
// next block, so that a wave's rows read adjacent blocks.

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    // Workgroups per row of the dispatch grid.
    stride: u32,
    // How many ranges the blocks are split into.
    splits: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;

const THREADS: u32 = 128u;
// Rows per thread, and per workgroup.
const PER: u32 = 8u;
const ROWS: u32 = THREADS * PER;
// Words per block: 28 bytes.
const WORDS: u32 = 7u;
// Bytes of trits per block.
const BYTES: u32 = 26u;

// The block's activations.
var<workgroup> xb: array<f32, 128>;
// For each byte position, the sum each byte value stands for.
var<workgroup> table: array<f32, 6656>;

// The element the `n`th trit of byte `m` of a block belongs to.
fn element(m: u32, n: u32) -> u32 {
    if m < 16u {
        return m + 16u * n;
    } else if m < 24u {
        return 80u + (m - 16u) + 8u * n;
    }
    return 120u + (m - 24u) + 2u * n;
}

// The trits of byte value `v`, most significant first, which come out of
// tripling it.
fn trits(v: u32) -> array<f32, 5> {
    var q = v;
    var t: array<f32, 5>;
    for (var n = 0u; n < 5u; n++) {
        t[n] = f32((q * 3u) >> 8u) - 1.0;
        q = (q * 3u) & 0xffu;
    }
    return t;
}

// Builds the tables for byte values `v` and `v + 128`, reading each byte
// position's elements once for both; the last two positions hold four
// trits.
fn build(v: u32) {
    let a = trits(v);
    let b = trits(v + THREADS);
    for (var m = 0u; m < BYTES; m++) {
        var sa = 0.0;
        var sb = 0.0;
        let count = select(5u, 4u, m >= 24u);
        for (var n = 0u; n < count; n++) {
            let e = xb[element(m, n)];
            sa = fma(a[n], e, sa);
            sb = fma(b[n], e, sb);
        }
        table[m * 256u + v] = sa;
        table[m * 256u + v + THREADS] = sb;
    }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let blocks = p.cols / 128u;
    let tiles = (p.rows + ROWS - 1u) / ROWS;
    let id = wg.y * p.stride + wg.x;
    let tile = id % tiles;
    let split = id / tiles;
    let per = (blocks + p.splits - 1u) / p.splits;
    let first = split * per;
    let last = min(blocks, first + per);
    let r0 = tile * ROWS + lid;

    var acc: array<f32, PER>;
    for (var b = first; b < last; b++) {
        // The rows' words, each row's clamped into the matrix so that
        // every load is in bounds; a row past the end sums to nothing
        // that is kept.
        var words: array<array<u32, WORDS>, PER>;
        for (var i = 0u; i < PER; i++) {
            let row = min(r0 + i * THREADS, p.rows - 1u);
            let base = (b * p.rows + row) * WORDS;
            for (var k = 0u; k < WORDS; k++) {
                words[i][k] = w[base + k];
            }
        }
        // The last block's readers must be done before the tables change.
        workgroupBarrier();
        xb[lid] = x[b * 128u + lid];
        workgroupBarrier();
        build(lid);
        workgroupBarrier();
        for (var i = 0u; i < PER; i++) {
            var sum = 0.0;
            for (var k = 0u; k < 6u; k++) {
                let word = words[i][k];
                sum += table[(4u * k) * 256u + (word & 0xffu)];
                sum += table[(4u * k + 1u) * 256u + ((word >> 8u) & 0xffu)];
                sum += table[(4u * k + 2u) * 256u + ((word >> 16u) & 0xffu)];
                sum += table[(4u * k + 3u) * 256u + (word >> 24u)];
            }
            let word = words[i][6u];
            sum += table[24u * 256u + (word & 0xffu)];
            sum += table[25u * 256u + ((word >> 8u) & 0xffu)];
            let scale = unpack2x16float(word).y;
            acc[i] = fma(scale, sum, acc[i]);
        }
    }
    for (var i = 0u; i < PER; i++) {
        let row = r0 + i * THREADS;
        if row < p.rows {
            if p.splits == 1u {
                out[row] = acc[i];
            } else {
                partial[split * p.rows + row] = acc[i];
            }
        }
    }
}
