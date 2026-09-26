// Causal attention over 8-bit caches, four integers to a word and a scale
// per block of 32 as a half-precision float, two to a word: one workgroup per (token, key/value head, chunk of 128 positions), for
// every query head that shares the key/value head, so that the caches are
// read once for the group and a long context spreads over the GPU. A
// subgroup scores one position at a time, its lanes each taking a few
// elements of the key and adding their products up across the subgroup,
// so that a key is read as one contiguous run; the scores are softmaxed
// against the chunk's own maximum; then each thread takes four elements of
// the head at every position of its lane and sums the values the scores
// weigh, and the lanes' sums are added together. The chunk's maximum, sum
// of exponentials, and weighted values go to the partials buffer for
// `attention_combine.wgsl` to merge, or, when there is only one chunk,
// straight to the output. A dispatch covers the tokens from `first` on,
// with `chunks` chunks for each.

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
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k_quants: array<u32>;
@group(0) @binding(3) var<storage, read> k_scales: array<u32>;
@group(0) @binding(4) var<storage, read> v_quants: array<u32>;
@group(0) @binding(5) var<storage, read> v_scales: array<u32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;

const CHUNK: u32 = 128u;

// Activations to a block of the caches, which share a scale.
const BLOCK: u32 = 32u;

// The four 8-bit integers of a word, lowest first, as floats.
fn bytes(w: u32) -> vec4<f32> {
    let shifted = vec4(w << 24u, w << 16u, w << 8u, w);
    return vec4<f32>(bitcast<vec4<i32>>(shifted) >> vec4(24u));
}

// The scale of block `b` of a cache, from the word of scales that holds it.
fn block_scale(scales: u32, b: u32) -> f32 {
    return unpack2x16float(scales >> (16u * (b & 1u))).x;
}

// Four activations of the key cache from `i` on, or the first two.
fn load_key(i: u32, two: bool) -> vec4<f32> {
    let w = k_quants[i / 4u] >> (8u * (i % 4u));
    let s = block_scale(k_scales[i / (2u * BLOCK)], i / BLOCK);
    let k = bytes(w) * s;
    return select(k, vec4(k.xy, 0.0, 0.0), two);
}

// The four activations of the value cache from `i` on.
fn load_value(i: u32) -> vec4<f32> {
    return bytes(v_quants[i / 4u]) * block_scale(v_scales[i / (2u * BLOCK)], i / BLOCK);
}

// Most query heads to a key/value head.
const GROUP: u32 = 8u;

var<workgroup> partial: array<array<f32, GROUP>, 16>;
var<workgroup> scores: array<array<f32, CHUNK>, GROUP>;
var<workgroup> values: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
    @builtin(subgroup_size) ssize: u32,
) {
    let c = wg.x % p.chunks;
    let tk = wg.x / p.chunks;
    let t = p.first + tk / p.n_kv_heads;
    let kvh = tk % p.n_kv_heads;
    let len = p.pos + t + 1u;
    let start = c * CHUNK;
    if start >= len {
        return;
    }
    let count = min(CHUNK, len - start);
    let group = p.n_heads / p.n_kv_heads;
    let kv = kvh * p.head_dim;
    let kv_dim = p.n_kv_heads * p.head_dim;
    let scale = inverseSqrt(f32(p.head_dim));
    // The first query head of the group, for this token.
    let qbase = (t * p.n_heads + kvh * group) * p.head_dim;

    // Each lane of a subgroup takes `per` elements of a key, two or four,
    // and keeps the queries' matching elements.
    let per = max(p.head_dim / ssize, 2u);
    let words = per / 2u;
    let in_head = sinv * per < p.head_dim;
    var query: array<vec4<f32>, GROUP>;
    for (var g = 0u; g < group; g++) {
        if in_head {
            let base = qbase + g * p.head_dim + sinv * per;
            query[g] = vec4(q[base], q[base + 1u], select(0.0, q[base + 2u], words > 1u), select(0.0, q[base + 3u], words > 1u));
        }
    }
    for (var i = sid; i < count; i += nsg) {
        var key = vec4(0.0);
        if in_head {
            key = load_key((start + i) * kv_dim + kv + sinv * per, words == 1u);
        }
        for (var g = 0u; g < group; g++) {
            let s = subgroupAdd(dot(query[g], key));
            if sinv == 0u {
                scores[g][i] = s * scale;
            }
        }
    }
    workgroupBarrier();

    // Each head's maximum over the chunk, for a stable softmax, and the
    // sum of its exponentials.
    var m: array<f32, GROUP>;
    var e: array<f32, GROUP>;
    for (var g = 0u; g < group; g++) {
        var s = -1e30;
        if lid < count {
            s = scores[g][lid];
        }
        let sm = subgroupMax(s);
        if sinv == 0u {
            partial[sid][g] = sm;
        }
    }
    workgroupBarrier();
    for (var g = 0u; g < group; g++) {
        m[g] = -1e30;
        for (var i = 0u; i < nsg; i++) {
            m[g] = max(m[g], partial[i][g]);
        }
        e[g] = 0.0;
        if lid < count {
            e[g] = exp(scores[g][lid] - m[g]);
        }
    }
    workgroupBarrier();
    var total: array<f32, GROUP>;
    for (var g = 0u; g < group; g++) {
        if lid < CHUNK {
            scores[g][lid] = e[g];
        }
        let sum = subgroupAdd(e[g]);
        if sinv == 0u {
            partial[sid][g] = sum;
        }
    }
    workgroupBarrier();
    for (var g = 0u; g < group; g++) {
        total[g] = 0.0;
        for (var i = 0u; i < nsg; i++) {
            total[g] += partial[i][g];
        }
    }

    // Each thread takes four elements of the head at every position of its
    // lane, the lanes striding through the positions together, so that a
    // position's values are read as one run and weighed for every head.
    let quads = p.head_dim / 4u;
    let lanes = 256u / quads;
    let d = lid % quads;
    let lane = lid / quads;
    var acc: array<vec4<f32>, GROUP>;
    if lane < lanes {
        for (var i = lane; i < count; i += lanes) {
            let v = load_value((start + i) * kv_dim + kv + 4u * d);
            for (var g = 0u; g < group; g++) {
                acc[g] += scores[g][i] * v;
            }
        }
    }
    for (var g = 0u; g < group; g++) {
        values[lid] = acc[g];
        workgroupBarrier();
        if lid < quads {
            var o = vec4(0.0);
            for (var l = 0u; l < lanes; l++) {
                o += values[l * quads + lid];
            }
            let h = kvh * group + g;
            let obase = (t * p.n_heads + h) * p.head_dim + 4u * lid;
            if p.chunks == 1u {
                o /= total[g];
                out[obase] = o.x;
                out[obase + 1u] = o.y;
                out[obase + 2u] = o.z;
                out[obase + 3u] = o.w;
            } else {
                let base = (((t - p.first) * p.n_heads + h) * p.chunks + c) * (p.head_dim + 2u);
                if lid == 0u {
                    partials[base] = m[g];
                    partials[base + 1u] = total[g];
                }
                partials[base + 2u + 4u * lid] = o.x;
                partials[base + 2u + 4u * lid + 1u] = o.y;
                partials[base + 2u + 4u * lid + 2u] = o.z;
                partials[base + 2u + 4u * lid + 3u] = o.w;
            }
        }
        workgroupBarrier();
    }
}
