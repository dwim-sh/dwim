// Causal attention with a bias by distance and a sink, over 8-bit caches
// that hold a ring of the latest positions: `attention.wgsl`, with what
// Inkling's attention adds to it. One workgroup per (token, key/value
// head, chunk of 128 positions), for every query head that shares the
// key/value head. A token attends over the positions no more than
// `window - 1` back, and the caches hold position `p` in row `p % ring`.
// A key `d` positions back has added to its score the dot product of the
// head's row of `rel` with row `d` of `bias`, if `bias` has one. The
// softmax is taken over the keys and a sink that has no value, whose score
// is the head's of `sinks`: here where a token has one chunk, and in
// `attention_rel_combine.wgsl` where it has more, from the chunks' maxima,
// sums of exponentials, and weighted values in the partials. A dispatch
// covers the tokens from `first` on, with `chunks` chunks for each.

struct Params {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
    window: u32,
    ring: u32,
    // How wide a head's row of `rel` is, and the distances `bias` has a
    // row for.
    n_rel: u32,
    extent: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k_quants: array<u32>;
@group(0) @binding(3) var<storage, read> k_scales: array<u32>;
@group(0) @binding(4) var<storage, read> v_quants: array<u32>;
@group(0) @binding(5) var<storage, read> v_scales: array<u32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;
@group(0) @binding(7) var<storage, read> rel: array<f32>;
@group(0) @binding(8) var<storage, read> bias: array<f32>;
@group(0) @binding(9) var<storage, read> sinks: array<f32>;

const CHUNK: u32 = 128u;

// Activations to a block of the caches, which share a scale.
const BLOCK: u32 = 32u;

// The four 8-bit integers of a word, lowest first, as floats.
fn bytes(w: u32) -> vec4<f32> {
    let shifted = vec4(w << 24u, w << 16u, w << 8u, w);
    return vec4<f32>(bitcast<vec4<i32>>(shifted) >> vec4(24u));
}

// The value of the bits of a non-negative half-precision float, by hand,
// since a driver may flush subnormal ones to zero and the scale of a block
// of small activations is one, as in `store_q8.wgsl`.
fn from_f16(h: u32) -> f32 {
    let e = h >> 10u;
    let m = h & 0x3ffu;
    if e == 0u {
        return f32(m) * bitcast<f32>(103u << 23u);
    }
    return bitcast<f32>(((e + 112u) << 23u) | (m << 13u));
}

// The scale of block `b` of a cache, from the word of scales that holds it.
fn block_scale(scales: u32, b: u32) -> f32 {
    return from_f16((scales >> (16u * (b & 1u))) & 0xffffu);
}

// The four activations of the key cache from `i` on.
fn load_key(i: u32) -> vec4<f32> {
    return bytes(k_quants[i / 4u]) * block_scale(k_scales[i / (2u * BLOCK)], i / BLOCK);
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
) {
    let c = wg.x % p.chunks;
    let tk = wg.x / p.chunks;
    let t = p.first + tk / p.n_kv_heads;
    let kvh = tk % p.n_kv_heads;
    // The token's own position, the first it attends over, and how many
    // it does.
    let at = p.pos + t;
    let lo = max(at + 1u, p.window) - p.window;
    let len = at + 1u - lo;
    if c * CHUNK >= len {
        return;
    }
    let start = lo + c * CHUNK;
    let count = min(CHUNK, len - c * CHUNK);
    let group = p.n_heads / p.n_kv_heads;
    let kv = kvh * p.head_dim;
    let kv_dim = p.n_kv_heads * p.head_dim;
    let scale = inverseSqrt(f32(p.head_dim));
    let qbase = (t * p.n_heads + kvh * group) * p.head_dim;

    // Each lane of a subgroup takes four elements of a key, the lanes side
    // by side, and keeps the queries' matching elements; a head wider than
    // the subgroup takes more than one round, each adding to the scores.
    // A subgroup scores the same positions in every round. The lanes are
    // counted rather than taken from `subgroup_size`, which is only the most
    // a subgroup holds.
    let quads = p.head_dim / 4u;
    let width = subgroupAdd(1u);
    let rounds = (quads + width - 1u) / width;
    for (var r = 0u; r < rounds; r++) {
        let d = r * width + sinv;
        let in_head = d < quads;
        var query: array<vec4<f32>, GROUP>;
        for (var g = 0u; g < group; g++) {
            if in_head {
                let base = qbase + g * p.head_dim + 4u * d;
                query[g] = vec4(q[base], q[base + 1u], q[base + 2u], q[base + 3u]);
            }
        }
        for (var i = sid; i < count; i += nsg) {
            var key = vec4(0.0);
            if in_head {
                key = load_key(((start + i) % p.ring) * kv_dim + kv + 4u * d);
            }
            for (var g = 0u; g < group; g++) {
                let s = subgroupAdd(dot(query[g], key)) * scale;
                if sinv == 0u {
                    if r == 0u {
                        scores[g][i] = s;
                    } else {
                        scores[g][i] += s;
                    }
                }
            }
        }
    }
    workgroupBarrier();

    // The bias for how far back each key is, a thread a position.
    var biased: array<f32, GROUP>;
    if lid < count {
        let distance = at - (start + lid);
        for (var g = 0u; g < group; g++) {
            var b = 0.0;
            if distance < p.extent {
                let row = (t * p.n_heads + kvh * group + g) * p.n_rel;
                for (var j = 0u; j < p.n_rel; j++) {
                    b += rel[row + j] * bias[distance * p.n_rel + j];
                }
            }
            biased[g] = scores[g][lid] + b;
        }
    }

    // Each head's maximum over the chunk, for a stable softmax, and the
    // sum of its exponentials.
    var m: array<f32, GROUP>;
    var e: array<f32, GROUP>;
    for (var g = 0u; g < group; g++) {
        var s = -1e30;
        if lid < count {
            s = biased[g];
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
            e[g] = exp(biased[g] - m[g]);
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
    let lanes = 256u / quads;
    let d = lid % quads;
    let lane = lid / quads;
    var acc: array<vec4<f32>, GROUP>;
    if lane < lanes {
        for (var i = lane; i < count; i += lanes) {
            let v = load_value(((start + i) % p.ring) * kv_dim + kv + 4u * d);
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
                // The sink is one more to the softmax: the sums are
                // rescaled to whichever is greater, the keys' maximum or
                // the sink.
                let sink = sinks[t * p.n_heads + h];
                let most = max(m[g], sink);
                let keys = exp(m[g] - most);
                o *= keys / (total[g] * keys + exp(sink - most));
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
