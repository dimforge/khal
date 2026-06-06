// Re-export a `Float` trait at crate root, backend-dependent.
// - non-nvptx (spirv/cpu): `spirv_std::num_traits::Float`.
// - nvptx + rust-cuda:     `cuda_std::float::GpuFloat` (hardware intrinsics).
// - nvptx + cuda-oxide:    a local trait whose methods call `core::intrinsics`
//   transcendentals; cuda-oxide lowers those to libdevice `__nv_*`. (libm
//   software math does not resolve under cuda-oxide's PTX backend.)
#[cfg(all(target_arch = "nvptx64", not(feature = "cuda-oxide")))]
pub use cuda_std::float::GpuFloat as Float;
#[cfg(not(target_arch = "nvptx64"))]
pub use spirv_std::num_traits::Float;

#[cfg(all(target_arch = "nvptx64", feature = "cuda-oxide"))]
pub use cuda_oxide_float::Float;

#[cfg(all(target_arch = "nvptx64", feature = "cuda-oxide"))]
mod cuda_oxide_float {
    /// Minimal `Float` for the cuda-oxide PTX backend. Methods lower to
    /// libdevice (`__nv_expf`, `__nv_sqrtf`, …) via `core::intrinsics`.
    pub trait Float: Copy {
        fn exp(self) -> Self;
        fn ln(self) -> Self;
        fn sqrt(self) -> Self;
        fn powf(self, n: Self) -> Self;
        fn abs(self) -> Self;
        fn floor(self) -> Self;
        fn ceil(self) -> Self;
        fn max(self, other: Self) -> Self;
        fn min(self, other: Self) -> Self;
        fn atan(self) -> Self;
    }

    macro_rules! float_impl {
        ($ty:ty, $exp:ident, $ln:ident, $sqrt:ident, $pow:ident, $floor:ident, $ceil:ident, $nvatan:ident) => {
            impl Float for $ty {
                #[inline(always)]
                fn exp(self) -> $ty { unsafe { core::intrinsics::$exp(self) } }
                #[inline(always)]
                fn ln(self) -> $ty { unsafe { core::intrinsics::$ln(self) } }
                #[inline(always)]
                fn sqrt(self) -> $ty { unsafe { core::intrinsics::$sqrt(self) } }
                #[inline(always)]
                fn powf(self, n: $ty) -> $ty { unsafe { core::intrinsics::$pow(self, n) } }
                #[inline(always)]
                fn floor(self) -> $ty { unsafe { core::intrinsics::$floor(self) } }
                #[inline(always)]
                fn ceil(self) -> $ty { unsafe { core::intrinsics::$ceil(self) } }
                // abs/max/min via plain ops (no stable intrinsic names here).
                #[inline(always)]
                fn abs(self) -> $ty { if self < 0.0 { -self } else { self } }
                #[inline(always)]
                fn max(self, other: $ty) -> $ty { if self >= other { self } else { other } }
                #[inline(always)]
                fn min(self, other: $ty) -> $ty { if self <= other { self } else { other } }
                #[inline(always)]
                fn atan(self) -> $ty {
                    unsafe extern "C" { fn $nvatan(x: $ty) -> $ty; }
                    unsafe { $nvatan(self) }
                }
            }
        };
    }
    float_impl!(f32, expf32, logf32, sqrtf32, powf32, floorf32, ceilf32, __nv_atanf);
    float_impl!(f64, expf64, logf64, sqrtf64, powf64, floorf64, ceilf64, __nv_atan);
}
