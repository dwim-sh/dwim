// Normalizes each row of x to unit length: one workgroup per row.

struct Params {
    dim: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

var<workgroup> partial: array<f32, 16>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let base = wg.x * p.dim;
    var ss = 0.0;
    for (var i = lid; i < p.dim; i += 256u) {
        let v = x[base + i];
        ss += v * v;
    }
    let s = subgroupAdd(ss);
    if sinv == 0u {
        partial[sid] = s;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < nsg; i++) {
        total += partial[i];
    }
    let scale = inverseSqrt(total + p.eps);
    for (var i = lid; i < p.dim; i += 256u) {
        x[base + i] *= scale;
    }
}
