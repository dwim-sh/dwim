//! Weights quantized as llama.cpp's `Q4_0` and `Q8_0` types are: blocks of
//! 32 values sharing one f16 scale, each value a 4-bit or an 8-bit integer
//! multiple of it. The devices keep a matrix's quants and its scales apart,
//! each row-major, so that a block's quants are whole 32-bit words (four
//! for `Q4_0`, eight for `Q8_0`) and its scale is half a word, rather than
//! llama.cpp's interleaving of an 18- or 34-byte block that leaves most of
//! them unaligned.
//!
//! A `Q4_0` block's 16 bytes hold element `j` in the low nibble of byte `j`
//! and element `j + 16` in its high nibble, each as a value from 0 to 15
//! that stands for 8 less; a `Q8_0` block's 32 bytes hold its elements in
//! order as signed integers.

use crate::{from_f16, quantize, to_f16};

/// Weights per block.
pub const BLOCK: usize = 32;

/// Blocks in a row of `cols` weights.
pub fn blocks(cols: usize) -> usize {
    assert!(
        cols.is_multiple_of(BLOCK),
        "rows must be a whole number of blocks"
    );
    cols / BLOCK
}

/// Quantizes a row of weights to `Q8_0`: each block's scale is its largest
/// magnitude over 127, and each weight the nearest multiple of it, as
/// [`quantize`] does for the caches.
pub fn quantize_q8_0_row(x: &[f32], scales: &mut [u16], quants: &mut [i8]) {
    assert_eq!(scales.len(), blocks(x.len()));
    assert_eq!(quants.len(), x.len());
    for ((block, scale), quants) in x
        .chunks_exact(BLOCK)
        .zip(scales)
        .zip(quants.chunks_exact_mut(BLOCK))
    {
        let (bits, q) = quantize(block);
        *scale = bits;
        quants.copy_from_slice(&q);
    }
}

/// Reconstructs a row of `Q8_0` weights.
pub fn dequantize_q8_0_row(scales: &[u16], quants: &[i8], out: &mut [f32]) {
    assert_eq!(scales.len(), blocks(out.len()));
    assert_eq!(quants.len(), out.len());
    for ((&scale, quants), out) in scales
        .iter()
        .zip(quants.chunks_exact(BLOCK))
        .zip(out.chunks_exact_mut(BLOCK))
    {
        let d = from_f16(scale);
        for (o, &q) in out.iter_mut().zip(quants) {
            *o = q as f32 * d;
        }
    }
}

/// Dot product of a row of `Q8_0` weights with activations.
pub fn dot_q8_0(scales: &[u16], quants: &[i8], x: &[f32]) -> f32 {
    assert_eq!(scales.len(), blocks(x.len()));
    assert_eq!(quants.len(), x.len());
    let mut acc = 0.0;
    for ((&scale, quants), x) in scales
        .iter()
        .zip(quants.chunks_exact(BLOCK))
        .zip(x.chunks_exact(BLOCK))
    {
        let mut sum = [0.0f32; 8];
        for (q, x) in quants.chunks_exact(8).zip(x.chunks_exact(8)) {
            for i in 0..8 {
                sum[i] += q[i] as f32 * x[i];
            }
        }
        acc += from_f16(scale) * sum.iter().sum::<f32>();
    }
    acc
}

/// Quantizes a row of weights to `Q4_0` as llama.cpp does: each block's
/// scale is the weight of largest magnitude, sign and all, over -8, so that
/// weight comes out as exactly -8 times the scale, and each weight rounds to
/// the nearest multiple of it from -8 to 7.
pub fn quantize_q4_0_row(x: &[f32], scales: &mut [u16], quants: &mut [u8]) {
    assert_eq!(scales.len(), blocks(x.len()));
    assert_eq!(quants.len(), x.len() / 2);
    for ((block, scale), quants) in x
        .chunks_exact(BLOCK)
        .zip(scales)
        .zip(quants.chunks_exact_mut(BLOCK / 2))
    {
        let max = block
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        *scale = to_f16(d);
        for (j, q) in quants.iter_mut().enumerate() {
            let lo = (block[j] * id + 8.5).clamp(0.0, 15.0) as u8;
            let hi = (block[j + BLOCK / 2] * id + 8.5).clamp(0.0, 15.0) as u8;
            *q = lo | (hi << 4);
        }
    }
}

/// Reconstructs a row of `Q4_0` weights.
pub fn dequantize_q4_0_row(scales: &[u16], quants: &[u8], out: &mut [f32]) {
    assert_eq!(scales.len(), blocks(out.len()));
    assert_eq!(quants.len(), out.len() / 2);
    for ((&scale, quants), out) in scales
        .iter()
        .zip(quants.chunks_exact(BLOCK / 2))
        .zip(out.chunks_exact_mut(BLOCK))
    {
        let d = from_f16(scale);
        for (j, &q) in quants.iter().enumerate() {
            out[j] = ((q & 0xf) as f32 - 8.0) * d;
            out[j + BLOCK / 2] = ((q >> 4) as f32 - 8.0) * d;
        }
    }
}

/// Dot product of a row of `Q4_0` weights with activations.
pub fn dot_q4_0(scales: &[u16], quants: &[u8], x: &[f32]) -> f32 {
    assert_eq!(scales.len(), blocks(x.len()));
    assert_eq!(quants.len(), x.len() / 2);
    let mut acc = 0.0;
    for ((&scale, quants), x) in scales
        .iter()
        .zip(quants.chunks_exact(BLOCK / 2))
        .zip(x.chunks_exact(BLOCK))
    {
        let (lo, hi) = x.split_at(BLOCK / 2);
        let mut sum = [0.0f32; 8];
        for ((q, lo), hi) in quants
            .chunks_exact(8)
            .zip(lo.chunks_exact(8))
            .zip(hi.chunks_exact(8))
        {
            for i in 0..8 {
                sum[i] += ((q[i] & 0xf) as f32 - 8.0) * lo[i] + ((q[i] >> 4) as f32 - 8.0) * hi[i];
            }
        }
        acc += from_f16(scale) * sum.iter().sum::<f32>();
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
    fn q8_0_round_trips_within_a_step() {
        let x = row(256, 1);
        let (mut scales, mut quants) = (vec![0; 8], vec![0; 256]);
        quantize_q8_0_row(&x, &mut scales, &mut quants);
        let mut back = vec![0.0; 256];
        dequantize_q8_0_row(&scales, &quants, &mut back);
        for (block, back) in x.chunks_exact(BLOCK).zip(back.chunks_exact(BLOCK)) {
            let step = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (a, b) in block.iter().zip(back) {
                assert!((a - b).abs() <= step * 0.51, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn q4_0_puts_the_largest_weight_at_minus_eight() {
        let mut x = vec![0.0f32; BLOCK];
        x[5] = -0.5;
        x[20] = 0.25;
        x[31] = 0.125;
        let (mut scales, mut quants) = (vec![0; 1], vec![0; 16]);
        quantize_q4_0_row(&x, &mut scales, &mut quants);
        // The scale is 0.0625: -0.5 is -8 of it, 0.25 is 4, 0.125 is 2.
        assert_eq!(from_f16(scales[0]), 0.0625);
        assert_eq!(quants[5] & 0xf, 0);
        assert_eq!(quants[4] >> 4, 12);
        assert_eq!(quants[15] >> 4, 10);
        assert_eq!(quants[0], 0x88);
        let mut back = vec![0.0; BLOCK];
        dequantize_q4_0_row(&scales, &quants, &mut back);
        assert_eq!(back, x);
    }

    #[test]
    fn dots_match_dequantized() {
        for cols in [32, 1024, 5120] {
            let w = row(cols, 2);
            let x = row(cols, 3);
            let mut back = vec![0.0; cols];

            let (mut scales, mut quants) = (vec![0; cols / BLOCK], vec![0; cols]);
            quantize_q8_0_row(&w, &mut scales, &mut quants);
            dequantize_q8_0_row(&scales, &quants, &mut back);
            let want: f32 = back.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (dot_q8_0(&scales, &quants, &x) - want).abs() < 1e-4 * cols as f32,
                "q8_0 cols {cols}"
            );

            let (mut scales, mut quants) = (vec![0; cols / BLOCK], vec![0; cols / 2]);
            quantize_q4_0_row(&w, &mut scales, &mut quants);
            dequantize_q4_0_row(&scales, &quants, &mut back);
            let want: f32 = back.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (dot_q4_0(&scales, &quants, &x) - want).abs() < 1e-4 * cols as f32,
                "q4_0 cols {cols}"
            );
        }
    }
}
