// A short causal convolution with a residual: each thread convolves one
// channel of one token over the token and the `taps - 1` before it,
// reading the ones before the batch from the state, the last tap for the
// token itself. With `heads`, the channels are split evenly between as
// many heads, each of which has `taps` more for the token in `dynamic`:
// the token `lag` back is also taken times the head's tap `lag`. The sum
// is added to the channel, and the whole divided by the square root of
// the taps and one. A token alone has its state moved on here, by the
// thread of each channel, once it has read what it needs of it; for a
// batch, `short_conv_state.wgsl` moves the state on afterwards.

struct Params {
    n: u32,
    channels: u32,
    taps: u32,
    // Heads with taps of their own, or none: `dynamic` is then not read.
    heads: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> dynamic: array<f32>;

// Channel c of the token `back` before token t, which is before the batch
// when further back than t.
fn input(t: u32, back: u32, c: u32) -> f32 {
    if back > t {
        return state[(p.taps - 1u + t - back) * p.channels + c];
    }
    return x[(t - back) * p.channels + c];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.n * p.channels {
        return;
    }
    let t = i / p.channels;
    let c = i % p.channels;
    var acc = x[i];
    for (var j = 0u; j < p.taps; j++) {
        acc += weight[c * p.taps + j] * input(t, p.taps - 1u - j, c);
    }
    var count = p.taps;
    if p.heads != 0u {
        let head = c / (p.channels / p.heads);
        for (var lag = 0u; lag < p.taps; lag++) {
            acc += dynamic[(t * p.heads + head) * p.taps + lag] * input(t, lag, c);
        }
        count = 2u * p.taps;
    }
    out[i] = acc * inverseSqrt(f32(count + 1u));
    // A convolution with one tap has no state to move on.
    if p.n == 1u && p.taps > 1u {
        for (var s = 0u; s + 2u < p.taps; s++) {
            state[s * p.channels + c] = state[(s + 1u) * p.channels + c];
        }
        state[(p.taps - 2u) * p.channels + c] = x[c];
    }
}
