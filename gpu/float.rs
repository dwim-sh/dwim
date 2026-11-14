//! The half-width floats weights and scales are kept as, as bits: bf16,
//! the top half of an f32, which the weights that are not quantized are
//! in, and IEEE half precision, which a block's scale is in.

/// Converts bf16 bits to f32: bf16 is the top half of an f32.
pub fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The largest finite half-precision float.
pub const F16_MAX: f32 = 65504.0;

/// Converts an f32 to the bits of the nearest IEEE half-precision float,
/// ties to even, clamped to the largest finite one.
pub fn to_f16(x: f32) -> u16 {
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let a = x.abs().min(F16_MAX);
    if a < f32::from_bits((127 - 14) << 23) {
        // Subnormal: multiples of 2^-24, which may round up to the smallest
        // normal, whose bits follow on from the largest subnormal's.
        return sign | (a * (1u32 << 24) as f32).round_ties_even() as u16;
    }
    let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    // The mantissa scaled to [1024, 2048), which rounding may carry to 2048.
    let m = (a * f32::from_bits(((127 + 10 - e) as u32) << 23)).round_ties_even() as u32;
    let (e, m) = if m == 2048 { (e + 1, 1024) } else { (e, m) };
    sign | (((e + 15) as u16) << 10) | (m - 1024) as u16
}

/// Rounds an f32 to the nearest bf16, ties to even: bf16 is the top half
/// of an f32.
pub fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// Converts the bits of a finite IEEE half-precision float to f32.
pub fn from_f16(bits: u16) -> f32 {
    let (bits, sign) = (bits as u32, bits as u32 & 0x8000);
    let exp = (bits >> 10) & 0x1f;
    // A normal float has an implicit leading one; a subnormal float scales
    // like the smallest normal one.
    let mantissa = (bits & 0x3ff) | (((exp != 0) as u32) << 10);
    let scale = f32::from_bits((exp.max(1) + 127 - 25) << 23);
    let magnitude = mantissa as f32 * scale;
    f32::from_bits(magnitude.to_bits() | (sign << 16))
}
