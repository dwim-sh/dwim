// The gated delta rule of linear attention: one workgroup per value head,
// stepping through the tokens in order. The head's state is a matrix of
// keys by values; each thread keeps a column of half of it in registers, so
// what the state holds for a key is summed over the two halves through
// shared memory, as is the output for the query.

struct Params {
    n: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    head_dim: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> gates: array<f32>;
@group(0) @binding(5) var<storage, read> decay: array<f32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;

// The head dimension the kernel is written for: 256 threads, each holding
// 64 of the 128 by 128 state.
const DIM: u32 = 128u;
const HALF: u32 = 64u;

var<workgroup> kv: array<f32, 256>;
var<workgroup> oq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wg.x;
    let j = lid % DIM;
    let half = lid / DIM;
    let kh = h / (p.n_v_heads / p.n_k_heads);
    let sbase = h * DIM * DIM;
    var s: array<vec4<f32>, 16>;
    for (var i = 0u; i < 16u; i++) {
        let row = half * HALF + 4u * i;
        s[i] = vec4(state[sbase + row * DIM + j], state[sbase + (row + 1u) * DIM + j],
                    state[sbase + (row + 2u) * DIM + j], state[sbase + (row + 3u) * DIM + j]);
    }
    let scale = inverseSqrt(f32(DIM));
    let a = decay[h];
    let dt_bias = decay[p.n_v_heads + h];
    for (var t = 0u; t < p.n; t++) {
        let alpha = gates[t * 2u * p.n_v_heads + h];
        let beta_raw = gates[t * 2u * p.n_v_heads + p.n_v_heads + h];
        let z = alpha + dt_bias;
        let softplus = select(log(1.0 + exp(z)), z, z > 20.0);
        let g = exp(a * softplus);
        let beta = 1.0 / (1.0 + exp(-beta_raw));
        // The state decays, and what it holds for the key is read out.
        let kbase = ((t * p.n_k_heads + kh) * DIM + half * HALF) / 4u;
        var mem = 0.0;
        for (var i = 0u; i < 16u; i++) {
            s[i] *= g;
            mem += dot(s[i], k[kbase + i]);
        }
        kv[lid] = mem;
        workgroupBarrier();
        let delta = (v[(t * p.n_v_heads + h) * DIM + j] - kv[j] - kv[DIM + j]) * beta;
        // The value takes its place, and the query reads the state out.
        let qbase = ((t * p.n_k_heads + kh) * DIM + half * HALF) / 4u;
        var o = 0.0;
        for (var i = 0u; i < 16u; i++) {
            s[i] += k[kbase + i] * delta;
            o += dot(s[i], q[qbase + i]);
        }
        oq[lid] = o;
        workgroupBarrier();
        if half == 0u {
            out[(t * p.n_v_heads + h) * DIM + j] = (oq[j] + oq[DIM + j]) * scale;
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 16u; i++) {
        let row = half * HALF + 4u * i;
        state[sbase + row * DIM + j] = s[i].x;
        state[sbase + (row + 1u) * DIM + j] = s[i].y;
        state[sbase + (row + 2u) * DIM + j] = s[i].z;
        state[sbase + (row + 3u) * DIM + j] = s[i].w;
    }
}
