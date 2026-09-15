// out[t] = table[tokens[t]]: each thread copies one pair of bf16 weights
// out of the embedding table as two activations.

struct Params {
    dim: u32,
    n: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> tokens: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.dim / 2u;
    let i = gid.x;
    if i >= p.n * half {
        return;
    }
    let t = i / half;
    let j = i % half;
    let pair = table[tokens[t] * half + j];
    out[t * p.dim + 2u * j] = bitcast<f32>(pair << 16u);
    out[t * p.dim + 2u * j + 1u] = bitcast<f32>(pair & 0xffff0000u);
}
