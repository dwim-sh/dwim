// out[k][t / 2] = (f16(x[t][k]), f16(x[t + 1][k])): the activations of a
// batch of tokens as half floats, two tokens to a word, by column, for
// `matmul_ternary_tile.wgsl` to stage a block of columns as one contiguous
// run. The pairs of a column are padded to a multiple of four with zeros,
// for the tile kernel to read them four at a time, and a batch of an odd
// number of tokens gets a zero token last.

struct Params {
    n: u32,
    cols: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;

// The bits of the half float nearest to `v`, ties to even: `pack2x16float`
// rounds toward zero on some drivers, which would pull every activation
// toward zero. A magnitude below the smallest normal half rounds to zero,
// and none is large enough to overflow.
fn half(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let sign = (bits >> 16u) & 0x8000u;
    let mag = bits & 0x7fffffffu;
    if mag < (113u << 23u) {
        return sign;
    }
    let rebased = mag - (112u << 23u);
    let rest = rebased & 0x1fffu;
    var h = rebased >> 13u;
    if rest > 0x1000u || (rest == 0x1000u && (h & 1u) == 1u) {
        h += 1u;
    }
    return sign | h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let tp = gid.y;
    let pairs = ((p.n + 1u) / 2u + 3u) / 4u * 4u;
    if k >= p.cols {
        return;
    }
    var a = 0.0;
    var b = 0.0;
    if 2u * tp < p.n {
        a = x[2u * tp * p.cols + k];
    }
    if 2u * tp + 1u < p.n {
        b = x[(2u * tp + 1u) * p.cols + k];
    }
    out[k * pairs + tp] = half(a) | (half(b) << 16u);
}
