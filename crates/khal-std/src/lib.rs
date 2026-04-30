//! GPU standard library for khal compute shaders.
//!
//! Provides cross-platform primitives (synchronization, atomics, indexing, iteration)
//! that compile to SPIR-V, CUDA PTX, and native CPU targets.

#![cfg_attr(any(target_arch = "spirv", target_arch = "nvptx64"), no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(link_llvm_intrinsics))]

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

/// Re-export of the `glamx` math library.
pub use glamx;

#[cfg(target_arch = "nvptx64")]
pub use cuda_std;


#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use std::println;
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
#[macro_export]
macro_rules! println {
    () => { };
    ($($arg:tt)*) => { };
}