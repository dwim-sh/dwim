// The draft's two-tap dynamic convolution: each thread takes one channel of
// one row, weighs it and the row before by the stage's base weights plus
// the row's dynamic weights for the channel's group, and adds the residual
// when finishing a sublayer's output. The row before the first is zero.

struct Params {
    rows: u32,
    hidden: u32,
    group: u32,
    stage: u32,
    // Whether the residual is added.
    residual: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dynamic: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.rows * p.hidden {
        return;
    }
    let t = i / p.hidden;
    let c = i % p.hidden;
    let groups = p.hidden / p.group;
    let g = c / p.group;
    let tap = 2u * p.stage;
    let dyn_base = t * 4u * groups;
    var value = x[i] * (base[tap * p.hidden + c] + dynamic[dyn_base + tap * groups + g]);
    if t > 0u {
        value += x[i - p.hidden] * (base[(tap + 1u) * p.hidden + c] + dynamic[dyn_base + (tap + 1u) * groups + g]);
    }
    if p.residual != 0u {
        value += residual[i];
    }
    out[i] = value;
}
