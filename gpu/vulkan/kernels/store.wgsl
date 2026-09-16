// cache[p.offset..][..p.len] = f16(src): each thread packs one pair of
// activations into a word of the cache as two half-precision floats, clamped
// to the largest finite one.

struct Params {
    // Both in activations, and even.
    offset: u32,
    len: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> cache: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.len / 2u {
        return;
    }
    let pair = vec2(src[2u * i], src[2u * i + 1u]);
    cache[p.offset / 2u + i] = pack2x16float(clamp(pair, vec2(-65504.0), vec2(65504.0)));
}
