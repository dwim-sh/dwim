// out = rotate(rmsnorm(x) * weight): one workgroup per 1024-element block
// of every row of x, which sums the squares of the whole row, then
// normalizes, scales, and signs its block in shared memory and transforms
// it there as ten rounds of butterflies, as `hadamard.wgsl` does.

struct Params {
    width: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> signs: array<f32>;

const BLOCK: u32 = 1024u;

var<workgroup> partial: array<f32, 16>;
var<workgroup> v: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_invocation_id) sinv: u32,
) {
    let blocks = p.width / BLOCK;
    let row = (wg.x / blocks) * p.width;
    let block = (wg.x % blocks) * BLOCK;
    var ss = 0.0;
    for (var i = lid; i < p.width; i += 256u) {
        let e = x[row + i];
        ss += e * e;
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
    for (var i = lid; i < BLOCK; i += 256u) {
        v[i] = x[row + block + i] * scale * weight[block + i] * signs[block + i];
    }
    workgroupBarrier();
    for (var h = 1u; h < BLOCK; h <<= 1u) {
        for (var pair = lid; pair < BLOCK / 2u; pair += 256u) {
            let i = (pair / h) * 2u * h + (pair % h);
            let a = v[i];
            let b = v[i + h];
            v[i] = a + b;
            v[i + h] = a - b;
        }
        workgroupBarrier();
    }
    for (var i = lid; i < BLOCK; i += 256u) {
        out[row + block + i] = v[i] * (1.0 / 32.0);
    }
}
