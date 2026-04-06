//! CPU- and CUDA-compatible wrappers for `spirv_std::arch` functions.
//!
//! On `target_arch = "spirv"`, these delegate to the real SPIR-V intrinsics.
//! On `target_arch = "nvptx64"`, these use LLVM NVVM intrinsics for CUDA.
//! On the CPU, they provide functional equivalents using `std::sync` primitives.
//!
//! Shader code should import from this module instead of `spirv_std::arch` directly
//! to enable CPU and CUDA execution.

mod atomics;
mod barriers;
mod subgroup;

pub use atomics::*;
pub use barriers::*;
pub use subgroup::*;
