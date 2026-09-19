// Normalizes each row of x by its root mean square and scales it by the
// weight: one workgroup per row. The row is read as vectors of four, a
// fixed number per thread with the index clamped rather than the load
// skipped, so that the loads go out together instead of one round trip
// after another.

struct Params {
    dim: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;

const THREADS: u32 = 256u;
// Vectors a thread reads: rows of up to 8192 elements.
const PER_THREAD: u32 = 8u;

var<workgroup> partial: array<f32, 16>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let vectors = p.dim / 4u;
    let base = wg.x * vectors;
    var vs: array<vec4<f32>, PER_THREAD>;
    var ss = 0.0;
    for (var k = 0u; k < PER_THREAD; k++) {
        let i = lid + k * THREADS;
        vs[k] = x[base + min(i, vectors - 1u)];
        ss += select(0.0, dot(vs[k], vs[k]), i < vectors);
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
    let scale = inverseSqrt(total / f32(p.dim) + p.eps);
    for (var k = 0u; k < PER_THREAD; k++) {
        let i = lid + k * THREADS;
        if i < vectors {
            x[base + i] = vs[k] * scale * weight[i];
        }
    }
}
