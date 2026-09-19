// out[i] = the sum over the column splits of `matmul_ternary_tile.wgsl`, or
// the block ranges of `matmul_ternary_sums.wgsl`, of their partial results,
// a thread per output: the batch's tokens by rows.

struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    stride: u32,
    splits: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> partial: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let size = p.n * p.rows;
    if i >= size {
        return;
    }
    var total = 0.0;
    for (var s = 0u; s < p.splits; s++) {
        total += partial[s * size + i];
    }
    out[i] = total;
}
