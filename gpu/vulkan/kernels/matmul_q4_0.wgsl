// out[t][r] = w[r] · x[t] for Q4_0 weights: as `matmul_q8_0.wgsl`, one
// workgroup per row and group of eight tokens, its threads taking a block
// of 32 weights each in turn. A block's quants are one 16-byte word: byte
// j holds element j in its low nibble and element j + 16 in its high one,
// each 8 more than the weight. The quants and the scales are laid out as
// `quant.rs` says.

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    // Workgroups per row of the dispatch grid, for matrices with more rows
    // than one dimension of the grid allows. The grid's second dimension
    // counts the groups of tokens of each such row.
    stride: u32,
    // Column splits, which only the tile kernel takes.
    splits: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;

// Tokens a workgroup takes.
const TOKENS: u32 = 8u;

var<workgroup> partial: array<array<f32, TOKENS>, 16>;

// The low nibbles of a word's four bytes, less eight, in memory order.
fn low(word: u32) -> vec4<f32> {
    return vec4<f32>(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
}

// The high nibbles, likewise.
fn high(word: u32) -> vec4<f32> {
    return vec4<f32>(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32(word >> 28u)) - 8.0;
}

// The scale of block i, half of a word of them.
fn scale(i: u32) -> f32 {
    let pair = unpack2x16float(scales[i / 2u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let token_groups = (p.n + TOKENS - 1u) / TOKENS;
    let r = (wg.y / token_groups) * p.stride + wg.x;
    let t0 = (wg.y % token_groups) * TOKENS;
    let count = min(TOKENS, p.n - t0);
    let blocks = p.cols / 32u;
    var acc: array<f32, TOKENS>;
    if r < p.rows {
        for (var b = lid; b < blocks; b += 64u) {
            let v = w[r * blocks + b];
            let d = scale(r * blocks + b);
            // Elements 0 to 15 from the low nibbles, 16 to 31 from the high.
            let q = array(low(v.x), low(v.y), low(v.z), low(v.w), high(v.x), high(v.y), high(v.z), high(v.w));
            for (var t = 0u; t < count; t++) {
                let xbase = ((t0 + t) * p.cols + b * 32u) / 4u;
                var sum = 0.0;
                for (var i = 0u; i < 8u; i++) {
                    sum += dot(q[i], x[xbase + i]);
                }
                acc[t] += d * sum;
            }
        }
    }
    for (var t = 0u; t < TOKENS; t++) {
        let s = subgroupAdd(acc[t]);
        if sinv == 0u {
            partial[sid][t] = s;
        }
    }
    workgroupBarrier();
    if lid < count && r < p.rows {
        var total = 0.0;
        for (var i = 0u; i < nsg; i++) {
            total += partial[i][lid];
        }
        out[(t0 + lid) * p.rows + r] = total;
    }
}
