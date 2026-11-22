// out[i][r] = scale[e] * (w[e][r] · x[i]) for pair i of a token and one of
// its experts, e, as `expert_matmul_nvfp4.wgsl` does, for Q8_0 weights:
// its threads take a block of 32 weights each in turn, the quants and the
// scales laid out as `quant.rs` says.

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
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> ids: array<f32>;
@group(0) @binding(5) var<storage, read> scale: array<f32>;

var<workgroup> partial: array<f32, 16>;

// The four signed bytes of a word, in memory order.
fn unpack(word: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(word << 24u) >> 24u),
        f32(i32(word << 16u) >> 24u),
        f32(i32(word << 8u) >> 24u),
        f32(i32(word) >> 24u),
    );
}

// The scale of block i, half of a word of them.
fn block_scale(i: u32) -> f32 {
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
    let r = wg.x;
    let pair = p.first + wg.y;
    let expert = u32(ids[pair]);
    let blocks = p.cols / 32u;
    let first = (expert * p.rows + r) * blocks;
    let xbase = select(pair, pair / p.k, p.per_token != 0u) * (p.cols / 4u);
    var acc = 0.0;
    for (var b = lid; b < blocks; b += 64u) {
        let lo = w[(first + b) * 2u];
        let hi = w[(first + b) * 2u + 1u];
        let at = xbase + b * 8u;
        let sum = dot(unpack(lo.x), x[at]) + dot(unpack(lo.y), x[at + 1u])
            + dot(unpack(lo.z), x[at + 2u]) + dot(unpack(lo.w), x[at + 3u])
            + dot(unpack(hi.x), x[at + 4u]) + dot(unpack(hi.y), x[at + 5u])
            + dot(unpack(hi.z), x[at + 6u]) + dot(unpack(hi.w), x[at + 7u]);
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
