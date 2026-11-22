// The SiTU activation: each row of gate_up is a gate then an up
// projection, each `width` wide, and a row of out gets
// beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta).

struct Params {
    len: u32,
    width: u32,
    beta: f32,
    linear_beta: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;

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
    let row = i / p.width;
    let j = i % p.width;
    let gate = gate_up[row * 2u * p.width + j];
    let up = gate_up[row * 2u * p.width + p.width + j];
    out[i] = p.beta * safe_tanh(gate / p.beta) / (1.0 + exp(-gate)) * p.linear_beta * safe_tanh(up / p.linear_beta);
}
