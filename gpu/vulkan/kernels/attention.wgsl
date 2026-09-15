// Causal attention over f16 caches, two half-precision floats to a word:
// one workgroup per (token, head). The threads first score the positions the
// token attends to, strided, into workgroup memory; then softmax the scores;
// then each thread sums one element of the head over the values, reading
// them coalesced.

struct Params {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
}

const MAX_LEN: u32 = 4096u;

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read> v_cache: array<u32>;

var<workgroup> scores: array<f32, MAX_LEN>;
var<workgroup> partial: array<f32, 16>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let t = wg.x / p.n_heads;
    let h = wg.x % p.n_heads;
    let group = p.n_heads / p.n_kv_heads;
    let kv = (h / group) * p.head_dim;
    let kv_dim = p.n_kv_heads * p.head_dim;
    let len = p.pos + t + 1u;
    let scale = inverseSqrt(f32(p.head_dim));
    let qbase = (t * p.n_heads + h) * p.head_dim;
    let quads = p.head_dim / 4u;

    // Scores, and their maximum for a stable softmax.
    var m = -1e30;
    for (var pos = lid; pos < len; pos += 256u) {
        var s = 0.0;
        let kbase = (pos * kv_dim + kv) / 2u;
        for (var d = 0u; d < quads; d++) {
            let k = vec4(unpack2x16float(k_cache[kbase + 2u * d]), unpack2x16float(k_cache[kbase + 2u * d + 1u]));
            s += dot(q[qbase / 4u + d], k);
        }
        s *= scale;
        scores[pos] = s;
        m = max(m, s);
    }
    m = subgroupMax(m);
    if sinv == 0u {
        partial[sid] = m;
    }
    workgroupBarrier();
    for (var i = 0u; i < nsg; i++) {
        m = max(m, partial[i]);
    }
    workgroupBarrier();

    var sum = 0.0;
    for (var pos = lid; pos < len; pos += 256u) {
        let e = exp(scores[pos] - m);
        scores[pos] = e;
        sum += e;
    }
    sum = subgroupAdd(sum);
    if sinv == 0u {
        partial[sid] = sum;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < nsg; i++) {
        total += partial[i];
    }

    // Each thread owns one element of the head.
    if lid < p.head_dim {
        var o = 0.0;
        for (var pos = 0u; pos < len; pos++) {
            let e = pos * kv_dim + kv + lid;
            o += scores[pos] * unpack2x16float(v_cache[e / 2u])[e % 2u];
        }
        out[qbase + lid] = o / total;
    }
}
