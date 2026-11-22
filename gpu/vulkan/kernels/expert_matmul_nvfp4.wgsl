// out[i][r] = scale[e] * (w[e][r] · x[i]) for pair i of a token and one of
// its experts, e: a matrix multiplication by a different matrix for each
// pair, out of the experts' matrices one after another in w. One workgroup
// per row and pair, its threads taking a block of 16 weights each in turn.
// The input is a row per token, shared by its experts, or a row per pair.
//
// The weights are NVFP4, laid out as `quant.rs` says: a block is 16
// weights in two words, byte j holding element j in its low nibble and
// element j + 8 in its high one, each a sign over three bits that stand
// for one of eight magnitudes, and its scale a byte, four to a word: a
// float of four bits of exponent over three of mantissa.

struct Params {
    rows: u32,
    cols: u32,
    // Experts per token, and whether the input is a row per token.
    k: u32,
    per_token: u32,
    // The pair the dispatch's first row of workgroups takes.
    first: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> ids: array<f32>;
@group(0) @binding(5) var<storage, read> scale: array<f32>;

var<workgroup> partial: array<f32, 16>;

// What four weights' bits stand for, doubled, as the scale is halved: the
// magnitudes 0, 1, 2, 3, 4, 6, 8 and 12 are four bits each of a word,
// the lowest first, and the fourth bit of a weight is its sign.
fn values(nibbles: vec4<u32>) -> vec4<f32> {
    let magnitude = vec4<f32>((vec4(0xc8643210u) >> (4u * (nibbles & vec4(7u)))) & vec4(0xfu));
    return select(magnitude, -magnitude, (nibbles & vec4(8u)) != vec4(0u));
}

// The low nibbles of a word's four bytes as weights, in memory order.
fn low(word: u32) -> vec4<f32> {
    return values(vec4(word, word >> 8u, word >> 16u, word >> 24u) & vec4(0xfu));
}

// The high nibbles, likewise.
fn high(word: u32) -> vec4<f32> {
    return values(vec4(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4(0xfu));
}

// The scale of block i, a byte of a word of them, halved. Zero and the
// bits of a NaN scale the block to nothing.
fn block_scale(i: u32) -> f32 {
    let bits = (scales[i / 4u] >> (8u * (i & 3u))) & 0xffu;
    if bits == 0u || bits == 0x7fu {
        return 0.0;
    }
    let e = (bits >> 3u) & 0xfu;
    let m = f32(bits & 7u);
    if e == 0u {
        return m * (0.5 / 512.0);
    }
    // (1 + m / 8) * 2^(e - 7) / 2, the power of two from its bits.
    return (8.0 + m) * bitcast<f32>((e + 116u) << 23u);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let r = wg.x;
    let pair = p.first + wg.y;
    let expert = u32(ids[pair]);
    let blocks = p.cols / 16u;
    let first = (expert * p.rows + r) * blocks;
    let xbase = select(pair, pair / p.k, p.per_token != 0u) * (p.cols / 4u);
    var acc = 0.0;
    for (var b = lid; b < blocks; b += 64u) {
        let v = w[first + b];
        let at = xbase + b * 4u;
        let sum = dot(low(v.x), x[at]) + dot(low(v.y), x[at + 1u]) + dot(high(v.x), x[at + 2u]) + dot(high(v.y), x[at + 3u]);
        acc += block_scale(first + b) * sum;
    }
    let s = subgroupAdd(acc);
    if sinv == 0u {
        partial[sid] = s;
    }
    workgroupBarrier();
    if lid == 0u {
        var total = 0.0;
        for (var i = 0u; i < nsg; i++) {
            total += partial[i];
        }
        out[pair * p.rows + r] = scale[expert] * total;
    }
}
