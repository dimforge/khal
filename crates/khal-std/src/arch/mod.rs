//! Architecture-specific runtime support modules.

/// CPU runtime: coroutine-based cooperative scheduling and parallel dispatch.
#[cfg(feature = "cpu")]
pub mod cpu;
/// CUDA intrinsics for thread and block indexing.
#[cfg(any(target_arch = "nvptx64", feature = "cuda-oxide"))]
pub mod cuda;
