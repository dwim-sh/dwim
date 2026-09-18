// Rotates each 1024-element block of every row of x by the normalized
// Walsh-Hadamard transform, multiplying by the signs before it (the forward
// rotation) or after it (the inverse): one workgroup per block, which it
// transforms in shared memory as ten rounds of butterflies.

struct Params {
    width: u32,
    inverse: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> signs: array<f32>;

const BLOCK: u32 = 1024u;

var<workgroup> v: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let blocks = p.width / BLOCK;
    let base = (wg.x / blocks) * p.width + (wg.x % blocks) * BLOCK;
    let sbase = (wg.x % blocks) * BLOCK;
    for (var i = lid; i < BLOCK; i += 256u) {
        var e = x[base + i];
        if p.inverse == 0u {
            e *= signs[sbase + i];
        }
        v[i] = e;
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
        var e = v[i] * (1.0 / 32.0);
        if p.inverse != 0u {
            e *= signs[sbase + i];
        }
        x[base + i] = e;
    }
}
