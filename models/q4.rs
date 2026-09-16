//! Four-bit weights: blocks of 32 values sharing one f16 scale, each value a
//! signed nibble of it, at 4.5 bits a weight. The same layout as llama.cpp's
//! `Q4_0`, so the two can be compared.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Weights per block.
pub const BLOCK: usize = 32;

/// Bytes per block: the scale, then 16 bytes of two nibbles each.
pub const BLOCK_BYTES: usize = 2 + BLOCK / 2;

/// Bytes of a quantized row of `cols` weights.
pub fn row_bytes(cols: usize) -> usize {
    assert!(cols.is_multiple_of(BLOCK), "rows must be a whole number of blocks");
    cols / BLOCK * BLOCK_BYTES
}

/// Quantizes a row of weights into `out`, which holds `row_bytes` for it.
pub fn quantize_row(x: &[f32], out: &mut [u8]) {
    assert_eq!(out.len(), row_bytes(x.len()));
    for (block, out) in x.chunks_exact(BLOCK).zip(out.chunks_exact_mut(BLOCK_BYTES)) {
        // The scale maps the value of largest magnitude to -8, so that the
        // nibbles' full range is used.
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[..2].copy_from_slice(&f16_from_f32(d).to_le_bytes());
        for j in 0..BLOCK / 2 {
            let lo = ((block[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let hi = ((block[j + BLOCK / 2] * id + 8.5) as i32).clamp(0, 15) as u8;
            out[2 + j] = lo | (hi << 4);
        }
    }
}

/// Reconstructs a quantized row's weights.
pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), row_bytes(out.len()));
    for (block, out) in row.chunks_exact(BLOCK_BYTES).zip(out.chunks_exact_mut(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for j in 0..BLOCK / 2 {
            out[j] = ((block[2 + j] & 0xf) as f32 - 8.0) * d;
            out[j + BLOCK / 2] = ((block[2 + j] >> 4) as f32 - 8.0) * d;
        }
    }
}

/// Dot product of a quantized row with activations.
pub fn dot(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len(), row_bytes(x.len()));
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { dot_avx2(row, x) };
    }
    dot_scalar(row, x)
}

fn dot_scalar(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0;
    for (block, x) in row.chunks_exact(BLOCK_BYTES).zip(x.chunks_exact(BLOCK)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut sum = 0.0;
        for j in 0..BLOCK / 2 {
            sum += ((block[2 + j] & 0xf) as f32 - 8.0) * x[j];
            sum += ((block[2 + j] >> 4) as f32 - 8.0) * x[j + BLOCK / 2];
        }
        acc += d * sum;
    }
    acc
}

/// Each block's nibbles are widened to floats eight at a time and multiplied
/// into the activations, with the block's scale applied once to its sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    unsafe {
        let mask = _mm_set1_epi8(0xf);
        let eight = _mm256_set1_epi32(8);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let blocks = x.len() / BLOCK;
        for b in 0..blocks {
            let block = row.as_ptr().add(b * BLOCK_BYTES);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let qs = _mm_loadu_si128(block.add(2) as *const __m128i);
            let lo = _mm_and_si128(qs, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask);
            let x = x.as_ptr().add(b * BLOCK);
            let f = |q: __m128i| _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q), eight));
            let mut sum = _mm256_mul_ps(f(lo), _mm256_loadu_ps(x));
            sum = _mm256_fmadd_ps(f(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(x.add(8)), sum);
            sum = _mm256_fmadd_ps(f(hi), _mm256_loadu_ps(x.add(16)), sum);
            sum = _mm256_fmadd_ps(f(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(x.add(24)), sum);
            if b % 2 == 0 {
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d), sum, acc0);
            } else {
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d), sum, acc1);
            }
        }
        let acc = _mm256_add_ps(acc0, acc1);
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum = _mm_add_ps(lo, hi);
        let sum = _mm_hadd_ps(sum, sum);
        let sum = _mm_hadd_ps(sum, sum);
        _mm_cvtss_f32(sum)
    }
}

/// Converts f32 to IEEE half precision, rounding to nearest even.
pub fn f16_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        // Infinity or NaN.
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let exp = exp - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        // Subnormal: shift the implicit one in.
        let mant = mant | 0x800000;
        let shift = (14 - exp) as u32;
        let half = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let round = (rem > (1 << (shift - 1))) || (rem == (1 << (shift - 1)) && half & 1 == 1);
        return sign | (half as u16 + round as u16);
    }
    let half = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let round = rem > 0x1000 || (rem == 0x1000 && half & 1 == 1);
    sign | (half as u16 + round as u16)
}

/// Converts IEEE half precision to f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f800000 | (mant << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_round_trips() {
        for v in [0.0, 1.0, -1.0, 0.5, 65504.0, 1e-5, 6.1e-5, 1.2345, -0.001, 1234.5] {
            let back = f16_to_f32(f16_from_f32(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
        assert_eq!(f16_from_f32(1.0), 0x3c00);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert!(f16_to_f32(f16_from_f32(f32::INFINITY)).is_infinite());
    }

    fn row(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn quantization_is_close() {
        let x = row(2048, 1);
        let mut q = vec![0; row_bytes(2048)];
        quantize_row(&x, &mut q);
        let mut back = vec![0.0; 2048];
        dequantize_row(&q, &mut back);
        let err: f32 = x.iter().zip(&back).map(|(a, b)| (a - b).abs()).sum::<f32>() / 2048.0;
        // A 4-bit grid over [-1, 1] has steps of about 1/8.
        assert!(err < 0.07, "mean error {err}");
        // The largest value in each block is exact.
        for (block, back) in x.chunks_exact(BLOCK).zip(back.chunks_exact(BLOCK)) {
            let i = (0..BLOCK).max_by(|&a, &b| block[a].abs().total_cmp(&block[b].abs())).unwrap();
            assert!((block[i] - back[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn dot_matches_dequantized() {
        for cols in [32, 512, 2048] {
            let w = row(cols, 2);
            let x = row(cols, 3);
            let mut q = vec![0; row_bytes(cols)];
            quantize_row(&w, &mut q);
            let mut back = vec![0.0; cols];
            dequantize_row(&q, &mut back);
            let want: f32 = back.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((dot(&q, &x) - want).abs() < 1e-3 * cols as f32, "cols {cols}");
            assert!((dot_scalar(&q, &x) - want).abs() < 1e-3 * cols as f32);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use rayon::prelude::*;
    use std::time::Instant;

    /// Throughput of the dot product over more weights than fit in cache,
    /// on every core: run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bandwidth() {
        let cols = 2048;
        let rows = 256 * 1024;
        let row = row_bytes(cols);
        let weights: Vec<u8> = (0..rows * row).map(|i| (i * 7 % 251) as u8).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.1).collect();
        let mut out = vec![0.0f32; rows];
        for _ in 0..2 {
            let start = Instant::now();
            out.par_chunks_mut(64)
                .zip(weights.par_chunks(64 * row))
                .for_each(|(out, rows)| {
                    for (o, r) in out.iter_mut().zip(rows.chunks_exact(row)) {
                        *o = dot(r, &x);
                    }
                });
            let secs = start.elapsed().as_secs_f64();
            println!(
                "{:.1} GB/s over {} MB ({:.1} G weights/s)",
                weights.len() as f64 / secs / 1e9,
                weights.len() >> 20,
                (rows * cols) as f64 / secs / 1e9
            );
        }
        let start = Instant::now();
        let mut acc = 0.0;
        for r in weights[..1024 * row].chunks_exact(row) {
            acc += dot(r, &x);
        }
        let secs = start.elapsed().as_secs_f64();
        println!("single core, in cache: {:.1} G weights/s ({acc})", (1024 * cols) as f64 / secs / 1e9);
    }
}
