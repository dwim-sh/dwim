// out = rotate(rmsnorm(x) * weight): one workgroup per 1024-element block
// of every row of x, which sums the squares of the whole row, then
// normalizes, scales, and signs its block and transforms it as
// `hadamard.wgsl` does: each thread holds four consecutive elements, the
// first two rounds of butterflies are between them, the next four between
// lanes of a subgroup by shuffles, and the last four between threads
// through shared memory. The row is read as vectors of four, a fixed
// number per thread with the index clamped rather than the load skipped,
// so that the loads go out together instead of one round trip after
// another.

struct Params {
    width: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> signs: array<vec4<f32>>;

const BLOCK: u32 = 1024u;
const THREADS: u32 = 256u;
// Vectors a thread reads of the row: rows of up to 8192 elements.
const PER_THREAD: u32 = 8u;

var<workgroup> partial: array<f32, 16>;
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
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let blocks = p.width / BLOCK;
    let vectors = p.width / 4u;
    let row = (wg.x / blocks) * vectors;
    let block = (wg.x % blocks) * (BLOCK / 4u);
    // The thread's vector of the block, its weight and its signs, loaded
    // along with the row for the sum, so that nothing waits on the sum but
    // the arithmetic.
    let mine = x[row + block + lid];
    let scaled = mine * weight[block + lid] * signs[block + lid];
    var ss = 0.0;
    for (var k = 0u; k < PER_THREAD; k++) {
        let i = lid + k * THREADS;
        let e = x[row + min(i, vectors - 1u)];
        ss += select(0.0, dot(e, e), i < vectors);
    }
    let s = subgroupAdd(ss);
    if sinv == 0u {
        partial[sid] = s;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < nsg; i++) {
        total += partial[i];
    }
    let scale = inverseSqrt(total / f32(p.width) + p.eps);
    out[row + block + lid] = butterfly(scaled * scale, lid, sinv) * (1.0 / 32.0);
}
