//! The block formats llama.cpp's quantizations ship in, as the file lays
//! them out, read into the formats the devices take. `Q4_0` and `Q8_0`
//! blocks are an f16 scale then their quants, 18 and 34 bytes; the devices
//! keep the same quants and scales apart, as `quant.rs` says. `Q4_1` blocks
//! add an f16 minimum after the scale, 20 bytes, and `Q6_K` blocks hold 256
//! weights in 210 bytes: the low four bits of each, the high two bits of
//! each packed four to a byte, sixteen 8-bit scales for sixteen weights
//! each, and an f16 scale for the block. The last two are not given to a
//! device: their weights are reconstructed and quantized again as `Q8_0`,
//! which keeps them to within a 254th of each block's largest magnitude,
//! finer than either format's own steps.

use dwim_gpu::{from_f16, quant};

use crate::Tensor;

/// Bytes of a file block of each format.
pub const Q4_0_BYTES: usize = 18;
pub const Q4_1_BYTES: usize = 20;
pub const Q8_0_BYTES: usize = 34;
pub const Q6_K_BYTES: usize = 210;

/// Weights in a `Q6_K` block.
pub const Q6_K_BLOCK: usize = 256;

fn f16(bytes: &[u8]) -> f32 {
    from_f16(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Reconstructs a row of `Q4_0` weights from the file.
pub fn dequantize_q4_0(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), out.len() / quant::BLOCK * Q4_0_BYTES);
    for (block, out) in row
        .chunks_exact(Q4_0_BYTES)
        .zip(out.chunks_exact_mut(quant::BLOCK))
    {
        let d = f16(block);
        for (j, &q) in block[2..].iter().enumerate() {
            out[j] = ((q & 0xf) as f32 - 8.0) * d;
            out[j + quant::BLOCK / 2] = ((q >> 4) as f32 - 8.0) * d;
        }
    }
}

/// Reconstructs a row of `Q4_1` weights from the file.
pub fn dequantize_q4_1(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), out.len() / quant::BLOCK * Q4_1_BYTES);
    for (block, out) in row
        .chunks_exact(Q4_1_BYTES)
        .zip(out.chunks_exact_mut(quant::BLOCK))
    {
        let d = f16(block);
        let m = f16(&block[2..]);
        for (j, &q) in block[4..].iter().enumerate() {
            out[j] = (q & 0xf) as f32 * d + m;
            out[j + quant::BLOCK / 2] = (q >> 4) as f32 * d + m;
        }
    }
}

/// Reconstructs a row of `Q8_0` weights from the file.
pub fn dequantize_q8_0(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), out.len() / quant::BLOCK * Q8_0_BYTES);
    for (block, out) in row
        .chunks_exact(Q8_0_BYTES)
        .zip(out.chunks_exact_mut(quant::BLOCK))
    {
        let d = f16(block);
        for (o, &q) in out.iter_mut().zip(&block[2..]) {
            *o = q as i8 as f32 * d;
        }
    }
}

/// Reconstructs a row of `Q6_K` weights from the file, as llama.cpp's
/// `dequantize_row_q6_K` does: each half of a block's 128 weights has its
/// low bits in 64 bytes and its high bits in 32, and the 32 weights
/// `l + 32k` of the half, for `k` from 0 to 3, come from nibble `k / 2` of
/// low byte `l + 32 (k % 2)` and bits `2k` and `2k + 1` of high byte `l`,
/// with scale `2k + l / 16` of the half's eight.
pub fn dequantize_q6_k(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), out.len() / Q6_K_BLOCK * Q6_K_BYTES);
    for (block, out) in row
        .chunks_exact(Q6_K_BYTES)
        .zip(out.chunks_exact_mut(Q6_K_BLOCK))
    {
        let (ql, rest) = block.split_at(128);
        let (qh, rest) = rest.split_at(64);
        let (scales, d) = rest.split_at(16);
        let d = f16(d);
        for half in 0..2 {
            let ql = &ql[half * 64..][..64];
            let qh = &qh[half * 32..][..32];
            let scales = &scales[half * 8..][..8];
            let out = &mut out[half * 128..][..128];
            for l in 0..32 {
                for k in 0..4 {
                    let low = ql[l + 32 * (k % 2)] >> (4 * (k / 2)) & 0xf;
                    let high = qh[l] >> (2 * k) & 3;
                    let q = (low | high << 4) as i32 - 32;
                    out[l + 32 * k] = d * scales[2 * k + l / 16] as i8 as f32 * q as f32;
                }
            }
        }
    }
}

/// A `Q4_0` matrix from the file, its quants and scales parted.
pub fn q4_0(data: &[u8], rows: usize, cols: usize) -> Tensor {
    let blocks = quant::blocks(cols);
    assert_eq!(data.len(), rows * blocks * Q4_0_BYTES);
    let mut scales = Vec::with_capacity(rows * blocks);
    let mut quants = Vec::with_capacity(rows * cols / 2);
    for block in data.chunks_exact(Q4_0_BYTES) {
        scales.push(u16::from_le_bytes([block[0], block[1]]));
        quants.extend_from_slice(&block[2..]);
    }
    Tensor::Q4_0 {
        shape: vec![rows, cols],
        scales,
        quants,
    }
}

/// A `Q8_0` matrix from the file, its quants and scales parted.
pub fn q8_0(data: &[u8], rows: usize, cols: usize) -> Tensor {
    let blocks = quant::blocks(cols);
    assert_eq!(data.len(), rows * blocks * Q8_0_BYTES);
    let mut scales = Vec::with_capacity(rows * blocks);
    let mut quants = Vec::with_capacity(rows * cols);
    for block in data.chunks_exact(Q8_0_BYTES) {
        scales.push(u16::from_le_bytes([block[0], block[1]]));
        quants.extend(block[2..].iter().map(|&q| q as i8));
    }
    Tensor::Q8_0 {
        shape: vec![rows, cols],
        scales,
        quants,
    }
}

/// A `Q8_0` matrix made of rows of weights, quantized again.
pub fn requantize_q8_0(rows: usize, cols: usize, mut row: impl FnMut(usize, &mut [f32])) -> Tensor {
    let blocks = quant::blocks(cols);
    let mut scales = vec![0; rows * blocks];
    let mut quants = vec![0; rows * cols];
    let mut values = vec![0.0; cols];
    for (r, (scales, quants)) in scales
        .chunks_exact_mut(blocks)
        .zip(quants.chunks_exact_mut(cols))
        .enumerate()
    {
        row(r, &mut values);
        quant::quantize_q8_0_row(&values, scales, quants);
    }
    Tensor::Q8_0 {
        shape: vec![rows, cols],
        scales,
        quants,
    }
}

/// Packs a row of `Q4_0` weights as the file lays them out, for tests.
#[cfg(any(test, feature = "testing"))]
pub fn pack_q4_0(x: &[f32]) -> Vec<u8> {
    let blocks = quant::blocks(x.len());
    let (mut scales, mut quants) = (vec![0; blocks], vec![0; x.len() / 2]);
    quant::quantize_q4_0_row(x, &mut scales, &mut quants);
    let mut out = Vec::with_capacity(blocks * Q4_0_BYTES);
    for (scale, quants) in scales.iter().zip(quants.chunks_exact(quant::BLOCK / 2)) {
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(quants);
    }
    out
}

/// Packs a row of `Q8_0` weights as the file lays them out, for tests.
#[cfg(any(test, feature = "testing"))]
pub fn pack_q8_0(x: &[f32]) -> Vec<u8> {
    let blocks = quant::blocks(x.len());
    let (mut scales, mut quants) = (vec![0; blocks], vec![0; x.len()]);
    quant::quantize_q8_0_row(x, &mut scales, &mut quants);
    let mut out = Vec::with_capacity(blocks * Q8_0_BYTES);
    for (scale, quants) in scales.iter().zip(quants.chunks_exact(quant::BLOCK)) {
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend(quants.iter().map(|&q| q as u8));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dwim_gpu::to_f16;

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
    fn packed_blocks_read_back_the_same_in_either_layout() {
        let x = row(256, 1);
        let mut from_file = vec![0.0; 256];
        let packed = pack_q4_0(&x);
        dequantize_q4_0(&packed, &mut from_file);
        assert_eq!(q4_0(&packed, 2, 128).dequantize(), from_file);
        assert!(x.iter().zip(&from_file).all(|(a, b)| (a - b).abs() < 0.15));
        let packed = pack_q8_0(&x);
        dequantize_q8_0(&packed, &mut from_file);
        assert_eq!(q8_0(&packed, 2, 128).dequantize(), from_file);
        assert!(x.iter().zip(&from_file).all(|(a, b)| (a - b).abs() < 0.005));
    }

    #[test]
    fn q4_1_adds_the_minimum() {
        // One block: scale 0.5, minimum -2, and nibbles 0 to 15 twice over.
        let mut block = Vec::new();
        block.extend_from_slice(&to_f16(0.5).to_le_bytes());
        block.extend_from_slice(&to_f16(-2.0).to_le_bytes());
        block.extend((0..16u8).map(|j| j | (15 - j) << 4));
        let mut out = vec![0.0; 32];
        dequantize_q4_1(&block, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], j as f32 * 0.5 - 2.0);
            assert_eq!(out[j + 16], (15 - j) as f32 * 0.5 - 2.0);
        }
    }

    #[test]
    fn q6_k_places_bits_and_scales_as_llama_cpp_does() {
        // A block of zeros but for weight 37 of the first half: l = 5,
        // k = 1, so its low nibble is the low one of byte 37, its high two
        // bits are bits 2 and 3 of high byte 5, and its scale is the third.
        let mut block = vec![0u8; Q6_K_BYTES];
        block[37] = 0b1010; // low nibble: 10
        block[128 + 5] = 0b1100; // bits 2 and 3: 3, so q = 58 - 32 = 26
        block[128 + 64 + 2] = 3; // scale: 3
        block[128 + 64 + 16..].copy_from_slice(&to_f16(0.25).to_le_bytes());
        let mut out = vec![0.0; Q6_K_BLOCK];
        dequantize_q6_k(&block, &mut out);
        assert_eq!(out[37], 0.25 * 3.0 * 26.0);
        // The scale's other fifteen weights, 32 to 47, have all bits clear,
        // which is -32; every other scale is zero.
        for (i, &v) in out.iter().enumerate() {
            let want = match i {
                37 => 19.5,
                32..48 => 0.25 * 3.0 * -32.0,
                _ => 0.0,
            };
            assert_eq!(v, want, "weight {i}");
        }
        // And the same in the second half, at weight 128 + 100: l = 4,
        // k = 3, high nibble of byte 64 + 36, bits 6 and 7 of high byte
        // 32 + 4, scale 8 + 6.
        let mut block = vec![0u8; Q6_K_BYTES];
        block[64 + 36] = 0b0111_0000;
        block[128 + 32 + 4] = 0b0100_0000;
        block[128 + 64 + 14] = 0xff; // -1
        block[128 + 64 + 16..].copy_from_slice(&to_f16(1.0).to_le_bytes());
        dequantize_q6_k(&block, &mut out);
        for (i, &v) in out.iter().enumerate() {
            let want = match i {
                228 => -(7.0 + 16.0 - 32.0),
                224..240 => -1.0 * -32.0,
                _ => 0.0,
            };
            assert_eq!(v, want, "weight {i}");
        }
    }

    #[test]
    fn requantizing_keeps_weights_within_a_step() {
        let x = row(2 * 64, 2);
        let t = requantize_q8_0(2, 64, |r, out| out.copy_from_slice(&x[r * 64..][..64]));
        for (a, b) in x.iter().zip(t.dequantize()) {
            assert!((a - b).abs() < 1.0 / 127.0, "{a} vs {b}");
        }
    }
}
