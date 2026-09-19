// Merges the chunks of `attention.wgsl`: one workgroup per (token, head),
// each thread taking one element of the head. The chunks' softmaxes were
// taken against their own maxima, so each is rescaled to the maximum over
// the chunks the token attends to before the sums are added.

struct Params {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> partials: array<f32>;

const CHUNK: u32 = 128u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = p.first + wg.x / p.n_heads;
    let h = wg.x % p.n_heads;
    if lid >= p.head_dim {
        return;
    }
    let len = p.pos + t + 1u;
    let chunks = (len + CHUNK - 1u) / CHUNK;
    let base = ((t - p.first) * p.n_heads + h) * p.chunks * (p.head_dim + 2u);
    let stride = p.head_dim + 2u;
    var m = -1e30;
    for (var c = 0u; c < chunks; c++) {
        m = max(m, partials[base + c * stride]);
    }
    var total = 0.0;
    var o = 0.0;
    for (var c = 0u; c < chunks; c++) {
        let scale = exp(partials[base + c * stride] - m);
        total += scale * partials[base + c * stride + 1u];
        o += scale * partials[base + c * stride + 2u + lid];
    }
    out[(t * p.n_heads + h) * p.head_dim + lid] = o / total;
}
