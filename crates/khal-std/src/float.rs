//! Floating-point conversion utilities.

use glamx::Vec2;

/// Unpack a u32 containing two f16 values into a Vec2 of f32.
///
/// The low 16 bits are the first f16 value, high 16 bits are the second.
///
/// On SPIR-V: uses `spirv_std::float::f16x2_to_vec2`.
/// On other targets: software conversion.
#[inline(always)]
pub fn unpack_half2x16(v: u32) -> Vec2 {
    #[cfg(target_arch = "spirv")]
    {
        let v2 = spirv_std::float::f16x2_to_vec2(v);
        Vec2::new(v2.x, v2.y)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let lo = (v & 0xFFFF) as u16;
        let hi = ((v >> 16) & 0xFFFF) as u16;
        Vec2::new(f16_to_f32(lo), f16_to_f32(hi))
    }
}

/// Convert an f16 (as u16 bits) to f32 in software.
#[cfg(not(target_arch = "spirv"))]
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            // Subnormal: normalize
            let mut m = mantissa;
            let mut e: i32 = -14 + 127;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            f32::from_bits(sign | ((e as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits(sign | 0x7F800000 | (mantissa << 13))
    } else {
        // Normal: exp is 1..30, so exp + 112 is 113..142
        let e = (exp as u32) + 112;
        f32::from_bits(sign | (e << 23) | (mantissa << 13))
    }
}
