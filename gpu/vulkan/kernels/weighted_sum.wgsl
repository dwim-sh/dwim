// out[t] = sum over e of weights[t][e] * y[t][e]: the outputs of a token's
// k experts, a row of `width` each, summed by what each counts for.

struct Params {
    len: u32,
    width: u32,
    k: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.len {
        return;
    }
    let t = i / p.width;
    let j = i % p.width;
    var acc = 0.0;
    for (var e = 0u; e < p.k; e++) {
        acc += weights[t * p.k + e] * y[(t * p.k + e) * p.width + j];
    }
    out[i] = acc;
}
