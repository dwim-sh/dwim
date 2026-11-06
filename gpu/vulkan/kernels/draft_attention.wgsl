// Attention of the draft's block of rows over the context before it, in a
// ring of slots per head, and the block itself: one workgroup per (row,
// query head), as `draft_attention.metal` does it. The threads score the
// keys between them, a whole key each, softmax the scores against their
// maximum, reduced through workgroup memory, then take an element of the
// head each at every key of their share and sum the values the scores
// weigh, the shares added up through workgroup memory.

struct Params {
    rows: u32,
    pos: u32,
    window: u32,
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k_ring: array<f32>;
@group(0) @binding(3) var<storage, read> v_ring: array<f32>;
@group(0) @binding(4) var<storage, read> k_block: array<f32>;
@group(0) @binding(5) var<storage, read> v_block: array<f32>;

const THREADS: u32 = 256u;
// Most keys a row sees: a window of 2048 and a block of up to 16 rows.
const MAX_KEYS: u32 = 2064u;

var<workgroup> scores: array<f32, MAX_KEYS>;
var<workgroup> reduce: array<f32, THREADS>;

// The key at index `i` of the row's keys: in the ring up to `context`, then
// in the block.
fn key_at(i: u32, context: u32, start: u32, kvh: u32, kv_dim: u32) -> u32 {
    if i < context {
        return (kvh * p.window + (start + i) % p.window) * p.head_dim;
    }
    return (i - context) * kv_dim + kvh * p.head_dim;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wg.x / p.n_heads;
    let head = wg.x % p.n_heads;
    let group = p.n_heads / p.n_kv_heads;
    let kvh = head / group;
    let kv_dim = p.n_kv_heads * p.head_dim;
    let position = p.pos + row;
    var start = 0u;
    if position + 1u > p.window {
        start = position + 1u - p.window;
    }
    let context = p.pos - start;
    let keys = context + p.rows;
    let qbase = (row * p.n_heads + head) * p.head_dim;
    let scale = inverseSqrt(f32(p.head_dim));

    for (var i = lid; i < keys; i += THREADS) {
        let at = key_at(i, context, start, kvh, kv_dim);
        var s = 0.0;
        for (var d = 0u; d < p.head_dim; d++) {
            if i < context {
                s = fma(q[qbase + d], k_ring[at + d], s);
            } else {
                s = fma(q[qbase + d], k_block[at + d], s);
            }
        }
        scores[i] = s * scale;
    }
    workgroupBarrier();
    var m = -3.0e38;
    for (var i = lid; i < keys; i += THREADS) {
        m = max(m, scores[i]);
    }
    reduce[lid] = m;
    workgroupBarrier();
    for (var s = THREADS / 2u; s > 0u; s >>= 1u) {
        if lid < s {
            reduce[lid] = max(reduce[lid], reduce[lid + s]);
        }
        workgroupBarrier();
    }
    m = reduce[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var i = lid; i < keys; i += THREADS) {
        let e = exp(scores[i] - m);
        scores[i] = e;
        sum += e;
    }
    reduce[lid] = sum;
    workgroupBarrier();
    for (var s = THREADS / 2u; s > 0u; s >>= 1u) {
        if lid < s {
            reduce[lid] += reduce[lid + s];
        }
        workgroupBarrier();
    }
    let total = reduce[0];
    workgroupBarrier();

    // Each thread takes element d of the head at every key of its share.
    let parts = THREADS / p.head_dim;
    let d = lid % p.head_dim;
    let part = lid / p.head_dim;
    var acc = 0.0;
    for (var i = part; i < keys; i += parts) {
        let at = key_at(i, context, start, kvh, kv_dim);
        if i < context {
            acc = fma(scores[i], v_ring[at + d], acc);
        } else {
            acc = fma(scores[i], v_block[at + d], acc);
        }
    }
    reduce[lid] = acc;
    workgroupBarrier();
    if part == 0u {
        for (var j = 1u; j < parts; j++) {
            acc += reduce[j * p.head_dim + d];
        }
        out[qbase + d] = acc / total;
    }
}
