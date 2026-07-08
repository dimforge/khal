//! GPU standard library for khal compute shaders.
//!
//! Provides cross-platform primitives (synchronization, atomics, indexing, iteration)
//! that compile to SPIR-V, CUDA PTX, and native CPU targets.

#![cfg_attr(any(target_arch = "spirv", target_arch = "nvptx64"), no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(link_llvm_intrinsics))]
// nvptx inline asm (warp shuffle in `sync::subgroup`) is unstable; the
// cuda-oxide backend compiles for the real nvptx64 target so it needs the gate.
#![cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch))]
// `Float` for the cuda-oxide backend calls `core::intrinsics::{expf32,..}`
// (lowered to libdevice `__nv_*` by cuda-oxide) instead of software libm.
#![cfg_attr(feature = "cuda-oxide", feature(core_intrinsics))]
#![cfg_attr(feature = "cuda-oxide", allow(internal_features))]

/// Architecture-specific runtime support (CPU coroutines, CUDA intrinsics).
pub mod arch;
/// Floating-point conversion utilities (f16, packing/unpacking).
pub mod float;
/// Indexing utilities with optional bounds-check removal.
pub mod index;
/// GPU-compatible iterators.
pub mod iter;
/// Re-exports of `spirv_std_macros` and `khal_derive::spirv_bindgen`.
pub mod macros;
/// Memory scope and semantics constants for SPIR-V and CUDA.
pub mod memory;
/// Numeric trait re-exports (`Float`) across backends.
pub mod num_traits;
/// Synchronization primitives (barriers, atomics).
pub mod sync;

/// Build-script helpers for shader crates. Host-only.
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub mod build_script;
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use build_script::*;

/// Re-export of the `glamx` math library.
pub use glamx;

#[cfg(all(target_arch = "nvptx64", not(feature = "cuda-oxide")))]
pub use cuda_std;
// NOTE: cuda-device is deliberately NOT re-exported at the crate root: a
// `pub use cuda_device;` gives its functions a `khal_std::cuda_device::*`
// visible path, which breaks the cuda-oxide importer's by-name intrinsic
// dispatch (it matches the canonical `cuda_device::*` paths). The types the
// generated entry code needs are re-exported from `cuda_oxide_glue` instead.

/// Glue for the cuda-oxide `spirv_bindgen` entry transform: wraps a
/// `&'static mut SharedArray` so a `#[spirv(workgroup)] &mut [T; N]` parameter
/// body (which calls `MaybeIndexUnchecked::read/write/at_mut`) lands on shared
/// memory.
#[cfg(all(feature = "cuda-oxide", not(target_arch = "spirv")))]
pub mod cuda_oxide_glue {
    use crate::index::MaybeIndexUnchecked;
    pub use cuda_device::SharedArray;

    pub struct SmemBuf<T: 'static, const N: usize>(pub &'static mut SharedArray<T, N>);
    impl<T: Copy, const N: usize> MaybeIndexUnchecked<T> for SmemBuf<T, N> {
        #[inline(always)]
        fn read(&self, i: usize) -> T { self.0[i] }
        #[inline(always)]
        fn write(&mut self, i: usize, v: T) { self.0[i] = v; }
        #[inline(always)]
        fn at_mut(&mut self, i: usize) -> &mut T { &mut self.0[i] }
        #[inline(always)]
        fn at(&self, i: usize) -> &T { &self.0[i] }
    }
}

#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use std::println;
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
#[macro_export]
macro_rules! println {
    () => {};
    ($($arg:tt)*) => {};
}
