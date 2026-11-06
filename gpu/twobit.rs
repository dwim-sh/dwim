//! Ternary weights two bits each: blocks of 128 values in {-1, 0, 1}
//! sharing one f16 scale, four to a byte, at 2.125 bits a weight. A block
//! is 34 bytes: the scale, then 32 bytes of codes, element `j` of the
//! block the two bits at `2 * (j % 4)` of byte `j / 4`, worth one less
//! than the code. The same layout as the `PQ2_0` type of PrismML's
//! llama.cpp, which Bonsai also ships in, with code 3 (worth 2) unused.
//!
//! The 1.75-bit packing of [`ternary`](crate::ternary) is what the model
//! holds and what the devices compute on; a file in this packing is
//! unpacked into it as it is read, which loses nothing, as both hold
//! trits. Holding the weights two bits each on the device was tried and
//! found slower: a token generated costs a read of every weight, and a
//! fifth more bytes is a fifth more time.

use rayon::prelude::*;

use crate::ternary::{self, BLOCK};

/// Bytes per block: the scale, then 32 of codes.
pub const BLOCK_BYTES: usize = 34;

/// Bytes of a quantized row of `cols` weights in a file.
pub fn row_bytes(cols: usize) -> usize {
    assert!(
        cols.is_multiple_of(BLOCK),
        "rows must be a whole number of blocks"
    );
    cols / BLOCK * BLOCK_BYTES
}

/// Unpacks a block of a file into its trits and its scale. A code of 3,
/// which no ternary weight takes, comes out as 2.
pub fn unpack(block: &[u8]) -> ([i8; BLOCK], f32) {
    assert_eq!(block.len(), BLOCK_BYTES);
    let mut trits = [0i8; BLOCK];
    for (j, t) in trits.iter_mut().enumerate() {
        *t = ((block[2 + j / 4] >> (2 * (j % 4))) & 3) as i8 - 1;
    }
    (
        trits,
        crate::from_f16(u16::from_le_bytes([block[0], block[1]])),
    )
}

/// Reconstructs a quantized row's weights.
pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), row_bytes(out.len()));
    for (block, out) in row
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK))
    {
        let (trits, d) = unpack(block);
        for (o, &t) in out.iter_mut().zip(&trits) {
            *o = t as f32 * d;
        }
    }
}

/// Repacks rows of a file, `rows` of `cols` each, into the 1.75-bit
/// packing of [`ternary`], or says which block holds a code of 3, which
/// is no ternary weight.
pub fn to_ternary(data: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    assert_eq!(data.len(), rows * row_bytes(cols));
    let mut out = vec![0u8; rows * ternary::row_bytes(cols)];
    let bad = out
        .par_chunks_exact_mut(ternary::row_bytes(cols))
        .zip(data.par_chunks_exact(row_bytes(cols)))
        .enumerate()
        .filter_map(|(r, (out, row))| {
            for (b, (out, block)) in out
                .chunks_exact_mut(ternary::BLOCK_BYTES)
                .zip(row.chunks_exact(BLOCK_BYTES))
                .enumerate()
            {
                if block[2..]
                    .iter()
                    .any(|&byte| (0..4).any(|n| (byte >> (2 * n)) & 3 == 3))
                {
                    return Some((r, b));
                }
                let (trits, _) = unpack(block);
                ternary::pack(&trits, u16::from_le_bytes([block[0], block[1]]), out);
            }
            None
        })
        .min();
    match bad {
        Some((r, b)) => Err(format!("block {b} of row {r} holds a code of 3")),
        None => Ok(out),
    }
}

/// Repacks rows of [`ternary`] blocks, `rows` of `cols` each, into the
/// file's packing: the inverse of [`to_ternary`].
pub fn from_ternary(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * ternary::row_bytes(cols));
    let mut out = Vec::with_capacity(rows * row_bytes(cols));
    for block in data.chunks_exact(ternary::BLOCK_BYTES) {
        let (trits, _) = ternary::unpack(block);
        out.extend_from_slice(&block[26..]);
        out.extend(trits.chunks_exact(4).map(codes));
    }
    out
}

/// The byte holding four trits as codes.
fn codes(four: &[i8]) -> u8 {
    four.iter()
        .enumerate()
        .map(|(n, &t)| ((t + 1) as u8) << (2 * n))
        .sum()
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

    /// A row in the file's packing, quantized as `ternary::quantize_row`
    /// does.
    fn quantize(x: &[f32]) -> Vec<u8> {
        let mut tern = vec![0; ternary::row_bytes(x.len())];
        ternary::quantize_row(x, &mut tern);
        from_ternary(&tern, 1, x.len())
    }

    #[test]
    fn matches_the_reference_layout() {
        // Element j is bits 2 (j % 4) of byte j / 4 after the scale.
        let mut x = vec![0.0f32; BLOCK];
        x[0] = 1.0;
        x[1] = -1.0;
        x[127] = 1.0;
        let q = quantize(&x);
        assert_eq!(q.len(), BLOCK_BYTES);
        assert_eq!(u16::from_le_bytes([q[0], q[1]]), crate::to_f16(1.0));
        assert_eq!(q[2], 0b01_01_00_10);
        assert_eq!(q[33], 0b10_01_01_01);
        let (trits, d) = unpack(&q);
        assert_eq!(d, 1.0);
        assert_eq!((trits[0], trits[1], trits[2], trits[127]), (1, -1, 0, 1));
    }

    #[test]
    fn dequantizes_and_repacks_like_ternary() {
        let x = row(1024, 5);
        let q = quantize(&x);
        let mut want = vec![0; ternary::row_bytes(1024)];
        ternary::quantize_row(&x, &mut want);
        assert_eq!(to_ternary(&q, 1, 1024).unwrap(), want);
        let mut a = vec![0.0; 1024];
        let mut b = vec![0.0; 1024];
        dequantize_row(&q, &mut a);
        ternary::dequantize_row(&want, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn refuses_a_code_of_three() {
        let mut q = quantize(&row(256, 6));
        q[BLOCK_BYTES + 2 + 7] |= 0b11_00_00_00;
        assert_eq!(
            to_ternary(&q, 1, 256).unwrap_err(),
            "block 1 of row 0 holds a code of 3"
        );
    }
}
