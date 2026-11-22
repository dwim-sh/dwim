// x = x - tanh(forget) * x + tanh(gain) * message: what a gate lets into
// the residual stream of a message, and what it has the stream forget.

struct Params {
    len: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> forget: array<f32>;
@group(0) @binding(2) var<storage, read> gain: array<f32>;
@group(0) @binding(3) var<storage, read> message: array<f32>;

// tanh, which a GPU's own may make not a number of where its exponentials
// overflow: past 15 either way it is one to the last bit.
fn safe_tanh(v: f32) -> f32 {
    return tanh(clamp(v, -15.0, 15.0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.len {
        return;
    }
    x[i] = x[i] - safe_tanh(forget[i]) * x[i] + safe_tanh(gain[i]) * message[i];
}
