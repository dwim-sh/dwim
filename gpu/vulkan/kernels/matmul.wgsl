// out[t][r] = w[r] · x[t]: one workgroup per row of the bf16 weight matrix,
// reading the row once as 16-byte chunks, eight weights at a time, and
// taking its dot product with each token's activations in turn.

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
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;

var<workgroup> partial: array<f32, 16>;

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
    let r = wg.y * p.stride + wg.x;
    let chunks = p.cols / 8u;
    let base = r * chunks;
    for (var t = 0u; t < p.n; t++) {
        var acc = 0.0;
        if r < p.rows {
            let xbase = t * (p.cols / 4u);
            for (var c = lid; c < chunks; c += 64u) {
                let wv = w[base + c];
                let x0 = x[xbase + c * 2u];
                let x1 = x[xbase + c * 2u + 1u];
                acc += lo(wv.x) * x0.x + hi(wv.x) * x0.y + lo(wv.y) * x0.z + hi(wv.y) * x0.w
                    + lo(wv.z) * x1.x + hi(wv.z) * x1.y + lo(wv.w) * x1.z + hi(wv.w) * x1.w;
            }
        }
        let s = subgroupAdd(acc);
        if sinv == 0u {
            partial[sid] = s;
        }
        workgroupBarrier();
        if lid == 0u && r < p.rows {
            var total = 0.0;
            for (var i = 0u; i < nsg; i++) {
                total += partial[i];
            }
            out[t * p.rows + r] = total;
        }
        workgroupBarrier();
    }
}
