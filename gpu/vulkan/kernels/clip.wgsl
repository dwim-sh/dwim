// Scales each row of x, `width` long, by `scale`, then each head of it,
// `dim` long, down to a length of `threshold` if it is longer: a thread
// per head.

struct Params {
    heads: u32,
    width: u32,
    dim: u32,
    threshold: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.heads {
        return;
    }
    let head = i * p.dim;
    let by = (i % (p.width / p.dim)) * p.dim;
    var square = 0.0;
    for (var j = 0u; j < p.dim; j++) {
        let v = x[head + j] * scale[by + j];
        square += v * v;
    }
    let factor = min(p.threshold / (sqrt(square) + 1e-6), 1.0);
    for (var j = 0u; j < p.dim; j++) {
        x[head + j] = x[head + j] * scale[by + j] * factor;
    }
}
