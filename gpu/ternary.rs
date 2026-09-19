//! Ternary weights: blocks of 128 values in {-1, 0, 1} sharing one f16
//! scale, packed five to a byte, at 1.75 bits a weight. The same layout as
//! the `PTQ1_0` type of PrismML's llama.cpp, which Bonsai ships in, so its
//! weights are used as they come.
//!
//! A block is 28 bytes: 24 bytes of five trits each, 2 bytes of four trits
//! each, and the scale. A byte holds its trits as a base-3 number scaled up
//! to fill the byte, so that multiplying by 3 leaves the most significant
//! trit in the top byte and the rest in the low one. Which element a trit
//! belongs to is not positional: byte `m` of the first 16 holds elements
//! `m + 16n`, byte `16 + m` of the next 8 holds elements `80 + m + 8n`, and
//! byte `24 + m` of the last two holds elements `120 + m + 2n`, where `n`
//! counts the trits of the byte from the most significant.

/// Weights per block.
pub const BLOCK: usize = 128;

/// Bytes per block: 26 of trits, then the scale.
pub const BLOCK_BYTES: usize = 28;

/// Bytes of a quantized row of `cols` weights.
pub fn row_bytes(cols: usize) -> usize {
    assert!(
        cols.is_multiple_of(BLOCK),
        "rows must be a whole number of blocks"
    );
    cols / BLOCK * BLOCK_BYTES
}

/// The element the `n`th trit of byte `m` of a block belongs to, and how
/// many trits the byte holds.
fn element(m: usize, n: usize) -> usize {
    match m {
        0..16 => m + 16 * n,
        16..24 => 80 + (m - 16) + 8 * n,
        _ => 120 + (m - 24) + 2 * n,
    }
}

fn trits_in(m: usize) -> usize {
    if m < 24 { 5 } else { 4 }
}

/// Quantizes a row of weights into `out`, which holds `row_bytes` for it:
/// each block's scale is its largest magnitude, and each weight rounds to
/// the nearest of -1, 0, and 1 times it.
pub fn quantize_row(x: &[f32], out: &mut [u8]) {
    assert_eq!(out.len(), row_bytes(x.len()));
    for (block, out) in x.chunks_exact(BLOCK).zip(out.chunks_exact_mut(BLOCK_BYTES)) {
        let d = block.iter().fold(0.0f32, |d, v| d.max(v.abs()));
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (m, byte) in out[..26].iter_mut().enumerate() {
            let mut q: u32 = 0;
            for n in 0..trits_in(m) {
                // -1, 0, 1 as 0, 1, 2.
                q = q * 3 + ((block[element(m, n)] * id).round() as i32 + 1) as u32;
            }
            if trits_in(m) == 4 {
                // Shift the first trit up to where five-trit bytes keep it.
                q *= 3;
            }
            // Scaled to fill the byte: 243 is 3 to the fifth.
            *byte = ((q * 256).div_ceil(243)) as u8;
        }
        out[26..].copy_from_slice(&crate::to_f16(d).to_le_bytes());
    }
}

/// Unpacks a block into its trits and its scale.
pub fn unpack(block: &[u8]) -> ([i8; BLOCK], f32) {
    assert_eq!(block.len(), BLOCK_BYTES);
    let mut trits = [0i8; BLOCK];
    for (m, &byte) in block[..26].iter().enumerate() {
        let mut q = byte as u32;
        for n in 0..trits_in(m) {
            trits[element(m, n)] = ((q * 3) >> 8) as i8 - 1;
            q = (q * 3) & 0xff;
        }
    }
    (
        trits,
        crate::from_f16(u16::from_le_bytes([block[26], block[27]])),
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

/// Dot product of a quantized row with activations.
pub fn dot(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len(), row_bytes(x.len()));
    let mut acc = 0.0;
    for (block, x) in row.chunks_exact(BLOCK_BYTES).zip(x.chunks_exact(BLOCK)) {
        let (trits, d) = unpack(block);
        let mut sum = [0.0f32; 8];
        for (t, x) in trits.chunks_exact(8).zip(x.chunks_exact(8)) {
            for i in 0..8 {
                sum[i] += t[i] as f32 * x[i];
            }
        }
        acc += d * sum.iter().sum::<f32>();
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
    fn every_element_is_covered_once() {
        let mut seen = [0; BLOCK];
        for m in 0..26 {
            for n in 0..trits_in(m) {
                seen[element(m, n)] += 1;
            }
        }
        assert!(seen.iter().all(|&s| s == 1));
    }

    #[test]
    fn round_trips_ternary_values() {
        // Weights that are already ternary come back exactly.
        let mut x: Vec<f32> = row(512, 1)
            .iter()
            .map(|v| (v * 1.5).round() * 0.25)
            .collect();
        x[0] = 0.25;
        let mut q = vec![0; row_bytes(512)];
        quantize_row(&x, &mut q);
        let mut back = vec![0.0; 512];
        dequantize_row(&q, &mut back);
        assert_eq!(x, back);
    }

    #[test]
    fn matches_the_reference_packing() {
        // A block laid out by hand: element m + 16n of the first sixteen
        // bytes is the nth trit of byte m, and so on.
        let mut x = vec![0.0f32; BLOCK];
        x[0] = 1.0; // byte 0, first trit: 2 * 81
        x[16] = -1.0; // byte 0, second trit: 0 * 27
        x[80] = 1.0; // byte 16, first trit
        x[121] = -1.0; // byte 25, first trit
        x[127] = 1.0; // byte 25, fourth trit
        let mut q = vec![0; BLOCK_BYTES];
        quantize_row(&x, &mut q);
        let code = |trits: &[u32]| -> u8 {
            let mut v = 0;
            for &t in trits {
                v = v * 3 + t;
            }
            ((v * 256).div_ceil(243)) as u8
        };
        assert_eq!(q[0], code(&[2, 0, 1, 1, 1]));
        assert_eq!(q[1], code(&[1, 1, 1, 1, 1]));
        assert_eq!(q[16], code(&[2, 1, 1, 1, 1]));
        assert_eq!(q[25], code(&[0, 1, 1, 2, 0]));
        assert_eq!(u16::from_le_bytes([q[26], q[27]]), crate::to_f16(1.0));
        let (trits, d) = unpack(&q);
        assert_eq!(d, 1.0);
        assert_eq!(trits[0], 1);
        assert_eq!(trits[16], -1);
        assert_eq!(trits[127], 1);
        assert_eq!(trits.iter().filter(|&&t| t != 0).count(), 5);
    }

    #[test]
    fn dot_matches_dequantized() {
        for cols in [128, 1024, 5120] {
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
}
