// Rotates each 1024-element block of every row of x by the normalized
// Walsh-Hadamard transform, multiplying by the signs before it (the forward
// rotation) or after it (the inverse): one workgroup per block. Each thread
// holds four consecutive elements: the first two rounds of butterflies are
// between them, the next four between lanes of a subgroup by shuffles, and
// the last four between threads through shared memory, each in one of two
// arrays in turn so that a round needs one barrier.

struct Params {
    width: u32,
    inverse: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> signs: array<vec4<f32>>;

const BLOCK: u32 = 1024u;
const THREADS: u32 = 256u;

var<workgroup> v: array<vec4<f32>, 2u * THREADS>;

// The Walsh-Hadamard transform of the block whose elements 4lid to 4lid + 3
// are `e`, unnormalized.
fn butterfly(e_in: vec4<f32>, lid: u32, sinv: u32) -> vec4<f32> {
    var e = e_in;
    e = vec4(e.x + e.y, e.x - e.y, e.z + e.w, e.z - e.w);
    e = vec4(e.x + e.z, e.y + e.w, e.x - e.z, e.y - e.w);
    // Elements 4s apart are in the lanes s apart; a subgroup has at least
    // 16 lanes.
    for (var s = 1u; s < 16u; s <<= 1u) {
        let o = vec4(subgroupShuffleXor(e.x, s), subgroupShuffleXor(e.y, s), subgroupShuffleXor(e.z, s), subgroupShuffleXor(e.w, s));
        e = select(o - e, e + o, (sinv & s) == 0u);
    }
    var side = 0u;
    for (var s = 16u; s < THREADS; s <<= 1u) {
        v[side + lid] = e;
        workgroupBarrier();
        let o = v[side + (lid ^ s)];
        e = select(o - e, e + o, (lid & s) == 0u);
        side = THREADS - side;
    }
    return e;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sinv: u32) {
    let blocks = p.width / BLOCK;
    let base = ((wg.x / blocks) * p.width + (wg.x % blocks) * BLOCK) / 4u + lid;
    let sbase = (wg.x % blocks) * BLOCK / 4u + lid;
    var e = x[base];
    if p.inverse == 0u {
        e *= signs[sbase];
    }
    e = butterfly(e, lid, sinv) * (1.0 / 32.0);
    if p.inverse != 0u {
        e *= signs[sbase];
    }
    x[base] = e;
}
