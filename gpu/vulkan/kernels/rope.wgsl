// Rotary position embeddings: each thread rotates one pair of elements,
// half the rotated part of a head apart, by the angle for the token's
// position.

struct Params {
    n_heads: u32,
    head_dim: u32,
    rot_dim: u32,
    pos: u32,
    n: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.rot_dim / 2u;
    let id = gid.x;
    if id >= p.n * p.n_heads * half {
        return;
    }
    let i = id % half;
    // Heads are numbered across tokens: token t's first is t * n_heads.
    let head = id / half;
    let t = head / p.n_heads;
    let base = head * p.head_dim;
    let angle = table[(p.pos + t) * half + i];
    let a = x[base + i];
    let b = x[base + i + half];
    x[base + i] = a * angle.x - b * angle.y;
    x[base + i + half] = b * angle.x + a * angle.y;
}
