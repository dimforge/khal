#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]
#![allow(clippy::result_large_err)]

/// GPU backend abstractions for WebGPU, CUDA, and CPU execution.
pub mod backend;

/// Typed GPU function dispatch and launch utilities.
pub mod function;
/// Shader trait definitions and argument binding.
pub mod shader;

pub use backend::{AsGpuSlice, AsGpuSliceMut};
#[cfg(feature = "derive")]
pub use khal_derive::*;
pub use shader::{Shader, ShaderArgs, ShaderArgsType};

/// Third-party modules re-exports.
pub mod re_exports {
    pub use bytemuck;
    pub use include_dir;
    pub use paste;
    #[cfg(feature = "webgpu")]
    pub use wgpu::{self, Device};
}

/// Re-export of [`backend::BufferUsages`] for convenience.
pub use backend::BufferUsages;
