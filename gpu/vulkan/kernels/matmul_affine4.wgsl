// out[t][r] = w[r] · x[t] for weights four bits each, in groups of 64 with
// a scale and a bias, as `affine4.rs` packs them: one workgroup per row and
// group of eight tokens, as `matmul_bf16.wgsl`, the threads reading the
// row's codes as chunks of 32 and taking each chunk's dot product with
// the eight tokens' activations, a group's weights entering as its scale
// times the sum of the codes' products plus its bias times the sum of the
// activations.

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
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;

// Tokens a workgroup takes.
const TOKENS: u32 = 8u;

var<workgroup> partial: array<array<f32, TOKENS>, 16>;

// The eight codes of a word, in memory order.
fn lo(word: u32) -> vec4<f32> {
    return vec4(f32(word & 15u), f32((word >> 4u) & 15u), f32((word >> 8u) & 15u), f32((word >> 12u) & 15u));
}

fn hi(word: u32) -> vec4<f32> {
    return vec4(f32((word >> 16u) & 15u), f32((word >> 20u) & 15u), f32((word >> 24u) & 15u), f32(word >> 28u));
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
    // A row's words: its codes, then a word per group of its scale and
    // bias as half floats.
    let row_words = p.cols / 8u + p.cols / 64u;
    let base = r * row_words;
    let scales = base + p.cols / 8u;
    let chunks = p.cols / 32u;
    var acc: array<f32, TOKENS>;
    if r < p.rows {
        for (var c = lid; c < chunks; c += 64u) {
            let sb = unpack2x16float(w[scales + c / 2u]);
            for (var t = 0u; t < count; t++) {
                let xbase = (t0 + t) * (p.cols / 4u) + c * 8u;
                var products = 0.0;
                var sum = 0.0;
                for (var k = 0u; k < 4u; k++) {
                    let word = w[base + c * 4u + k];
                    let x0 = x[xbase + 2u * k];
                    let x1 = x[xbase + 2u * k + 1u];
                    products += dot(lo(word), x0) + dot(hi(word), x1);
                    sum += dot(x0, vec4(1.0)) + dot(x1, vec4(1.0));
                }
                acc[t] += sb.x * products + sb.y * sum;
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
