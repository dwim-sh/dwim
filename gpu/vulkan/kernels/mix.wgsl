// Attention over the depth: out[t] is a mix of token t's rows of up to
// six states, each normalized by its root mean square, weighted by the
// softmax over the states of each normalized row's dot product with the
// query, over the square root of the width. One workgroup per token: its
// threads take four elements each of every state in turn, the sums of
// squares and the dot products are added up across the group, and each
// thread writes its elements of the mix.

struct Params {
    dim: u32,
    // States there are: the bindings past them are not read.
    count: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s0: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> s1: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> s2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> s3: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> s4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> s5: array<vec4<f32>>;

const STATES: u32 = 6u;

var<workgroup> partial: array<array<f32, 12>, 16>;

// Vector `at` of state s.
fn load(s: u32, at: u32) -> vec4<f32> {
    switch s {
        case 0u { return s0[at]; }
        case 1u { return s1[at]; }
        case 2u { return s2[at]; }
        case 3u { return s3[at]; }
        case 4u { return s4[at]; }
        default { return s5[at]; }
    }
}

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
    for (var s = 0u; s < p.count; s++) {
        var squares = 0.0;
        var product = 0.0;
        for (var i = lid; i < vectors; i += 256u) {
            let v = load(s, base + i);
            squares += dot(v, v);
            product += dot(v, query[i]);
        }
        let all_squares = subgroupAdd(squares);
        let all_products = subgroupAdd(product);
        if sinv == 0u {
            partial[sid][2u * s] = all_squares;
            partial[sid][2u * s + 1u] = all_products;
        }
    }
    workgroupBarrier();

    // What normalizes each state, and its score against the query; then
    // each state's share of the mix, times what normalizes it.
    var share: array<f32, STATES>;
    var norms: array<f32, STATES>;
    var most = -1e30;
    for (var s = 0u; s < p.count; s++) {
        var squares = 0.0;
        var product = 0.0;
        for (var i = 0u; i < nsg; i++) {
            squares += partial[i][2u * s];
            product += partial[i][2u * s + 1u];
        }
        norms[s] = inverseSqrt(squares / f32(p.dim) + p.eps);
        share[s] = product * norms[s] * inverseSqrt(f32(p.dim));
        most = max(most, share[s]);
    }
    var sum = 0.0;
    for (var s = 0u; s < p.count; s++) {
        share[s] = exp(share[s] - most);
        sum += share[s];
    }
    for (var s = 0u; s < p.count; s++) {
        share[s] = share[s] / sum * norms[s];
    }
    for (var i = lid; i < vectors; i += 256u) {
        var o = vec4(0.0);
        for (var s = 0u; s < p.count; s++) {
            o += share[s] * load(s, base + i);
        }
        out[base + i] = o;
    }
}
