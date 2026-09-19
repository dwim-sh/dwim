// out[t][r] = w[r] · x[t]: one workgroup per row of the bf16 weight matrix
// and group of eight tokens, reading the row once as 16-byte chunks, eight
// weights at a time, and taking each chunk's dot product with the eight
// tokens' activations, so that a batch of tokens costs the row's reads
// once and no more barriers than one token does.

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
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;

// Tokens a workgroup takes.
const TOKENS: u32 = 8u;

var<workgroup> partial: array<array<f32, TOKENS>, 16>;

fn lo(pair: u32) -> f32 {
    return bitcast<f32>(pair << 16u);
}

fn hi(pair: u32) -> f32 {
    return bitcast<f32>(pair & 0xffff0000u);
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
    let chunks = p.cols / 8u;
    let base = r * chunks;
    var acc: array<f32, TOKENS>;
    if r < p.rows {
        for (var c = lid; c < chunks; c += 64u) {
            let wv = w[base + c];
            let ws = array(lo(wv.x), hi(wv.x), lo(wv.y), hi(wv.y), lo(wv.z), hi(wv.z), lo(wv.w), hi(wv.w));
            for (var t = 0u; t < count; t++) {
                let xbase = (t0 + t) * (p.cols / 4u) + c * 2u;
                let x0 = x[xbase];
                let x1 = x[xbase + 1u];
                acc[t] += ws[0] * x0.x + ws[1] * x0.y + ws[2] * x0.z + ws[3] * x0.w + ws[4] * x1.x + ws[5] * x1.y + ws[6] * x1.z + ws[7] * x1.w;
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
