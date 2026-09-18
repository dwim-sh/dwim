// The causal convolution of linear attention: each thread convolves one
// channel of one token over the last four tokens, reading the ones before
// the batch from the state, passes the result through SiLU, and writes it to
// the q, k, or v buffer the channel belongs to. The threads of the first
// token also save the last three tokens' inputs as the next state.

struct Params {
    n: u32,
    channels: u32,
    q_dim: u32,
    k_dim: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<f32>;
@group(0) @binding(6) var<storage, read> weight: array<f32>;

const KERNEL: i32 = 4;

// Channel c of the token at `t`, which is before the batch when negative.
fn input(t: i32, c: u32) -> f32 {
    if t < 0 {
        return state[u32(t + KERNEL - 1) * p.channels + c];
    }
    return x[u32(t) * p.channels + c];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.n * p.channels {
        return;
    }
    let t = i / p.channels;
    let c = i % p.channels;
    var acc = 0.0;
    for (var j = 0; j < KERNEL; j++) {
        acc += weight[c * u32(KERNEL) + u32(j)] * input(i32(t) + j + 1 - KERNEL, c);
    }
    let y = acc / (1.0 + exp(-acc));
    let v_dim = p.channels - p.q_dim - p.k_dim;
    if c < p.q_dim {
        q[t * p.q_dim + c] = y;
    } else if c < p.q_dim + p.k_dim {
        k[t * p.k_dim + c - p.q_dim] = y;
    } else {
        v[t * v_dim + c - p.q_dim - p.k_dim] = y;
    }
    if t == 0u {
        for (var s = 0; s < KERNEL - 1; s++) {
            state_out[u32(s) * p.channels + c] = input(i32(p.n) + s + 1 - KERNEL, c);
        }
    }
}
