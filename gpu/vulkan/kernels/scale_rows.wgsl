// Scales each row of x, `width` long, by its factor.

struct Params {
    len: u32,
    width: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> factors: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.len {
        return;
    }
    x[i] *= factors[i / p.width];
}
