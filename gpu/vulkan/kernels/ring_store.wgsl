// Stores rows of keys or values into a ring of slots per head: each thread
// takes one element of one row, and puts it in the slot of the row's
// position of its head.

struct Params {
    rows: u32,
    pos: u32,
    window: u32,
    n_kv_heads: u32,
    head_dim: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> ring: array<f32>;
@group(0) @binding(1) var<storage, read> rows: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let kv_dim = p.n_kv_heads * p.head_dim;
    if i >= p.rows * kv_dim {
        return;
    }
    let t = i / kv_dim;
    let h = (i % kv_dim) / p.head_dim;
    let d = i % p.head_dim;
    let slot = (p.pos + t) % p.window;
    ring[(h * p.window + slot) * p.head_dim + d] = rows[i];
}
