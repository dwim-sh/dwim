// A gate on a message: x = x - tanh(forget) * x + tanh(gain) * message,
// where the gain and what is forgotten are projections of a few
// dimensions, which are themselves projections of x and of the message,
// each normalized by its root mean square. One workgroup per token: its
// threads add up the squares of both rows; each subgroup then takes rows
// of the projections down in turn, its lanes eight weights of both
// matrices each at a time; and each thread takes elements of x four at a
// time, with the rows of the two projections back up for them. The
// weights are bf16, eight to a vector of words, the low half of a word
// first.

struct Params {
    dim: u32,
    rank: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> message: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> down_input: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> down_message: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> gain: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> forget: array<vec4<u32>>;

// Most dimensions the gate goes through.
const RANK: u32 = 64u;

var<workgroup> partial: array<array<f32, 2>, 16>;
var<workgroup> down: array<f32, RANK>;

// The four bf16 weights in two words, in memory order.
fn weights(pairs: vec2<u32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(vec4(pairs.x << 16u, pairs.x & 0xffff0000u, pairs.y << 16u, pairs.y & 0xffff0000u));
}

// tanh, which a GPU's own may make not a number of where its exponentials
// overflow: past 15 either way it is one to the last bit.
fn safe_tanh(v: f32) -> f32 {
    return tanh(clamp(v, -15.0, 15.0));
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let vectors = p.dim / 4u;
    let base = wg.x * vectors;

    // The squares of both rows, added up across the group.
    var squares = 0.0;
    var message_squares = 0.0;
    for (var i = lid; i < vectors; i += 256u) {
        let v = x[base + i];
        let m = message[base + i];
        squares += dot(v, v);
        message_squares += dot(m, m);
    }
    let all_squares = subgroupAdd(squares);
    let all_message_squares = subgroupAdd(message_squares);
    if sinv == 0u {
        partial[sid][0] = all_squares;
        partial[sid][1] = all_message_squares;
    }
    workgroupBarrier();
    squares = 0.0;
    message_squares = 0.0;
    for (var i = 0u; i < nsg; i++) {
        squares += partial[i][0];
        message_squares += partial[i][1];
    }
    let norm = inverseSqrt(squares / f32(p.dim) + p.eps);
    let message_norm = inverseSqrt(message_squares / f32(p.dim) + p.eps);

    // Down to the gate's few dimensions: a row of each projection a
    // subgroup at a time, eight weights a lane. The lanes are counted, as
    // a subgroup may hold fewer than its size says.
    let chunks = p.dim / 8u;
    let width = subgroupAdd(1u);
    for (var r = sid; r < p.rank; r += nsg) {
        var from_input = 0.0;
        var from_message = 0.0;
        for (var c = sinv; c < chunks; c += width) {
            let wi = down_input[r * chunks + c];
            let wm = down_message[r * chunks + c];
            from_input += dot(weights(wi.xy), x[base + 2u * c]) + dot(weights(wi.zw), x[base + 2u * c + 1u]);
            from_message += dot(weights(wm.xy), message[base + 2u * c]) + dot(weights(wm.zw), message[base + 2u * c + 1u]);
        }
        let sum = subgroupAdd(from_input) * norm + subgroupAdd(from_message) * message_norm;
        if sinv == 0u {
            down[r] = sum;
        }
    }
    workgroupBarrier();

    // Back up, four elements of x a thread at a time: their rows of the
    // two projections against the few dimensions.
    let per_row = p.rank / 8u;
    for (var i = lid; i < vectors; i += 256u) {
        var g = vec4(0.0);
        var f = vec4(0.0);
        for (var e = 0u; e < 4u; e++) {
            for (var c = 0u; c < per_row; c++) {
                let wgain = gain[(4u * i + e) * per_row + c];
                let wforget = forget[(4u * i + e) * per_row + c];
                let lo = vec4(down[8u * c], down[8u * c + 1u], down[8u * c + 2u], down[8u * c + 3u]);
                let hi = vec4(down[8u * c + 4u], down[8u * c + 5u], down[8u * c + 6u], down[8u * c + 7u]);
                g[e] += dot(weights(wgain.xy), lo) + dot(weights(wgain.zw), hi);
                f[e] += dot(weights(wforget.xy), lo) + dot(weights(wforget.zw), hi);
            }
        }
        let v = x[base + i];
        let tf = vec4(safe_tanh(f.x), safe_tanh(f.y), safe_tanh(f.z), safe_tanh(f.w));
        let tg = vec4(safe_tanh(g.x), safe_tanh(g.y), safe_tanh(g.z), safe_tanh(g.w));
        x[base + i] = v - tf * v + tg * message[base + i];
    }
}
