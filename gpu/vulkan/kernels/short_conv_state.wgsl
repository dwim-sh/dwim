// Moves the state of a short convolution on past a batch: the rows of the
// `taps - 1` tokens before the next, the last of the batch, or of the
// state as it was where the batch is shorter. A thread per channel reads
// what it needs of the old state before it writes the new.

struct Params {
    n: u32,
    channels: u32,
    taps: u32,
    heads: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;

// Most taps a convolution has, less one.
const KEPT: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= p.channels {
        return;
    }
    var next: array<f32, KEPT>;
    for (var s = 0u; s + 1u < p.taps; s++) {
        if p.n + s + 1u >= p.taps {
            next[s] = x[(p.n + s + 1u - p.taps) * p.channels + c];
        } else {
            next[s] = state[(p.n + s) * p.channels + c];
        }
    }
    for (var s = 0u; s + 1u < p.taps; s++) {
        state[s * p.channels + c] = next[s];
    }
}
