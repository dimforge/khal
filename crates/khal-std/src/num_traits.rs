// Re-export num_traits at crate root.
// On non-nvptx64: from spirv_std. On nvptx64: a compat module that re-exports
// cuda_std::GpuFloat as Float so `use crate::num_traits::Float` works.
/// On nvptx64, re-export `cuda_std::GpuFloat` as `Float` so that
/// `use crate::num_traits::Float` works the same as `spirv_std::num_traits::Float`.
#[cfg(target_arch = "nvptx64")]
pub use cuda_std::float::GpuFloat as Float;
#[cfg(not(target_arch = "nvptx64"))]
pub use spirv_std::num_traits::Float;
