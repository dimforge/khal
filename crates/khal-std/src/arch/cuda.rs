//! CUDA intrinsic wrappers for thread and block indexing.
//!
//! Two backends provide the underlying `%tid`/`%ctaid`/`%ntid`/`%nctaid`
//! special-register reads:
//! - `rust-cuda` (default): `cuda_std::thread` (Rust-CUDA / rustc_codegen_nvvm).
//! - `cuda-oxide`: `cuda_device::thread::*` accessors, recognized by name by
//!   the cuda-oxide MIR importer. `cuda_std` does not build under cuda-oxide.

use glamx::UVec3;

#[cfg(not(feature = "cuda-oxide"))]
use cuda_std::thread;

/// Special-register reads for the `cuda-oxide` backend, routed through
/// `cuda_device::thread::*`: the cuda-oxide MIR importer recognizes those
/// paths by name on ANY compilation target, so this works both for unified
/// (host-target) shader builds and for `--target nvptx64` device builds.
/// Each lowers to a single `mov.u32 %r, %sreg` in PTX.
#[cfg(feature = "cuda-oxide")]
mod thread {
    macro_rules! sreg {
        ($name:ident, $dev:ident) => {
            #[inline(always)]
            pub fn $name() -> u32 {
                cuda_device::thread::$dev()
            }
        };
    }
    sreg!(thread_idx_x, threadIdx_x);
    sreg!(thread_idx_y, threadIdx_y);
    sreg!(thread_idx_z, threadIdx_z);
    sreg!(block_idx_x, blockIdx_x);
    sreg!(block_idx_y, blockIdx_y);
    sreg!(block_idx_z, blockIdx_z);
    sreg!(block_dim_x, blockDim_x);
    sreg!(block_dim_y, blockDim_y);
    sreg!(block_dim_z, blockDim_z);
    sreg!(grid_dim_x, gridDim_x);
    sreg!(grid_dim_y, gridDim_y);
    sreg!(grid_dim_z, gridDim_z);
}

/// Returns the thread index within the current block as a `UVec3`.
#[inline(always)]
pub fn thread_idx() -> UVec3 {
    UVec3::new(
        thread::thread_idx_x() as u32,
        thread::thread_idx_y() as u32,
        thread::thread_idx_z() as u32,
    )
}

/// Returns the block index within the grid as a `UVec3`.
#[inline(always)]
pub fn block_idx() -> UVec3 {
    UVec3::new(
        thread::block_idx_x() as u32,
        thread::block_idx_y() as u32,
        thread::block_idx_z() as u32,
    )
}

/// Returns the block dimensions (threads per block) as a `UVec3`.
#[inline(always)]
pub fn block_dim() -> UVec3 {
    UVec3::new(
        thread::block_dim_x() as u32,
        thread::block_dim_y() as u32,
        thread::block_dim_z() as u32,
    )
}

/// Returns the global invocation ID (`block_idx * block_dim + thread_idx`).
#[inline(always)]
pub fn global_invocation_id() -> UVec3 {
    block_idx() * block_dim() + thread_idx()
}

/// Returns the local invocation ID (alias for [`thread_idx`]).
#[inline(always)]
pub fn local_invocation_id() -> UVec3 {
    thread_idx()
}

/// Returns the workgroup ID (alias for [`block_idx`]).
#[inline(always)]
pub fn workgroup_id() -> UVec3 {
    block_idx()
}

/// Returns the number of workgroups (grid dimensions) as a `UVec3`.
#[inline(always)]
pub fn num_workgroups() -> UVec3 {
    UVec3::new(
        thread::grid_dim_x(),
        thread::grid_dim_y(),
        thread::grid_dim_z(),
    )
}
