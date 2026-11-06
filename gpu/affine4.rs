//! Weights four bits each: groups of 64 values on a line, `scale * code +
//! bias`, with the group's scale and bias as f16 and its codes 0 to 15
//! packed two to a byte, at 4.5 bits a weight. The same rounding as MLX's
//! affine quantization and as Splash gives a draft's projections, which
//! the draft's makers trained it to stand. A row holds all its codes
//! first, a group's 32 bytes after another's, then the groups' scales and
//! biases, a pair of f16 each, so that a kernel reads a step of codes in
//! whole vectors.

use rayon::prelude::*;

/// Weights per group.
pub const GROUP: usize = 64;

/// Bytes of a quantized row of `cols` weights.
pub fn row_bytes(cols: usize) -> usize {
    assert!(
        cols.is_multiple_of(GROUP),
        "rows must be a whole number of groups"
    );
    cols / 2 + cols / GROUP * 4
}

/// Quantizes a row of weights into `out`, which holds `row_bytes` for it.
pub fn quantize_row(x: &[f32], out: &mut [u8]) {
    let cols = x.len();
    assert_eq!(out.len(), row_bytes(cols));
    let (codes, scales) = out.split_at_mut(cols / 2);
    for (g, group) in x.chunks_exact(GROUP).enumerate() {
        let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // The scale and bias as the kernels read them, rounded to f16, so
        // that the codes are chosen against the values they reconstruct.
        let scale = crate::from_f16(crate::to_f16((max - min) / 15.0));
        let bias = crate::from_f16(crate::to_f16(min));
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (byte, pair) in codes[g * GROUP / 2..][..GROUP / 2]
            .iter_mut()
            .zip(group.chunks_exact(2))
        {
            let code = |v: f32| (((v - bias) * inverse).round().clamp(0.0, 15.0)) as u8;
            *byte = code(pair[0]) | code(pair[1]) << 4;
        }
        scales[4 * g..][..2].copy_from_slice(&crate::to_f16(scale).to_le_bytes());
        scales[4 * g + 2..][..2].copy_from_slice(&crate::to_f16(bias).to_le_bytes());
    }
}

/// Quantizes a bf16 matrix of `rows` by `cols`, given as its bits.
pub fn quantize_bf16(bits: &[u16], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(bits.len(), rows * cols);
    let mut out = vec![0u8; rows * row_bytes(cols)];
    out.par_chunks_exact_mut(row_bytes(cols))
        .zip(bits.par_chunks_exact(cols))
        .for_each(|(out, row)| {
            let floats: Vec<f32> = row.iter().map(|&b| crate::bf16(b)).collect();
            quantize_row(&floats, out);
        });
    out
}

/// The scale and bias of group `g` of a row of `cols`.
fn group(row: &[u8], cols: usize, g: usize) -> (f32, f32) {
    let at = cols / 2 + 4 * g;
    (
        crate::from_f16(u16::from_le_bytes([row[at], row[at + 1]])),
        crate::from_f16(u16::from_le_bytes([row[at + 2], row[at + 3]])),
    )
}

/// Reconstructs a quantized row's weights.
pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    let cols = out.len();
    assert_eq!(row.len(), row_bytes(cols));
    for (g, out) in out.chunks_exact_mut(GROUP).enumerate() {
        let (scale, bias) = group(row, cols, g);
        for (i, o) in out.iter_mut().enumerate() {
            let byte = row[g * GROUP / 2 + i / 2];
            let code = (byte >> (4 * (i % 2))) & 15;
            *o = scale * code as f32 + bias;
        }
    }
}

/// Dot product of a quantized row with activations.
pub fn dot(row: &[u8], x: &[f32]) -> f32 {
    let cols = x.len();
    assert_eq!(row.len(), row_bytes(cols));
    let mut acc = 0.0;
    for (g, x) in x.chunks_exact(GROUP).enumerate() {
        let (scale, bias) = group(row, cols, g);
        let mut codes = 0.0;
        let mut sum = 0.0;
        for (i, &v) in x.iter().enumerate() {
            let byte = row[g * GROUP / 2 + i / 2];
            let code = (byte >> (4 * (i % 2))) & 15;
            codes += code as f32 * v;
            sum += v;
        }
        acc += scale * codes + bias * sum;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn round_trips_within_a_step() {
        let x = row(512, 1);
        let mut q = vec![0; row_bytes(512)];
        quantize_row(&x, &mut q);
        let mut back = vec![0.0; 512];
        dequantize_row(&q, &mut back);
        for (a, b) in x.iter().zip(&back) {
            // A step is at most 2 / 15 of the range, half of it either way.
            assert!((a - b).abs() <= 0.07, "{a} vs {b}");
        }
        // The extremes of a group come back as they went.
        let max = x[..64].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let at = x[..64].iter().position(|&v| v == max).unwrap();
        assert!((back[at] - max).abs() < 1e-3);
    }

    #[test]
    fn dot_matches_dequantized() {
        for cols in [64, 1024, 5120] {
            let w = row(cols, 2);
            let x = row(cols, 3);
            let mut q = vec![0; row_bytes(cols)];
            quantize_row(&w, &mut q);
            let mut back = vec![0.0; cols];
            dequantize_row(&q, &mut back);
            let want: f32 = back.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (dot(&q, &x) - want).abs() < 1e-3 * cols as f32,
                "cols {cols}"
            );
        }
    }

    #[test]
    fn quantizes_bf16_rows_alike() {
        let w = row(256, 4);
        let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let floats: Vec<f32> = bits.iter().map(|&b| crate::bf16(b)).collect();
        let mut want = vec![0; 2 * row_bytes(128)];
        quantize_row(&floats[..128], &mut want[..row_bytes(128)]);
        quantize_row(&floats[128..], &mut want[row_bytes(128)..]);
        assert_eq!(quantize_bf16(&bits, 2, 128), want);
    }
}
