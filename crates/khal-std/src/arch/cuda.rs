//! CUDA intrinsic wrappers for thread and block indexing.
//!
//! Two backends provide the underlying `%tid`/`%ctaid`/`%ntid`/`%nctaid`
//! special-register reads:
//! - `rust-cuda` (default): `cuda_std::thread` (Rust-CUDA / rustc_codegen_nvvm).
//! - `cuda-oxide`: khal_std's own `llvm.nvvm.read.ptx.sreg.*` externs, lowered
//!   by the cuda-oxide PTX backend. `cuda_std` does not build under cuda-oxide.

use glamx::UVec3;

#[cfg(not(feature = "cuda-oxide"))]
use cuda_std::thread;

/// khal_std's own nvptx special-register reads, used by the `cuda-oxide`
/// backend in place of `cuda_std::thread`. Each lowers to a single
/// `mov.u32 %r, %sreg` in PTX.
#[cfg(feature = "cuda-oxide")]
mod thread {
    macro_rules! sreg {
        ($name:ident, $intr:literal) => {
            #[inline(always)]
            pub fn $name() -> u32 {
                unsafe extern "C" {
                    #[link_name = $intr]
                    fn f() -> u32;
                }
                unsafe { f() }
            }
        };
    }
    sreg!(thread_idx_x, "llvm.nvvm.read.ptx.sreg.tid.x");
    sreg!(thread_idx_y, "llvm.nvvm.read.ptx.sreg.tid.y");
    sreg!(thread_idx_z, "llvm.nvvm.read.ptx.sreg.tid.z");
    sreg!(block_idx_x, "llvm.nvvm.read.ptx.sreg.ctaid.x");
    sreg!(block_idx_y, "llvm.nvvm.read.ptx.sreg.ctaid.y");
    sreg!(block_idx_z, "llvm.nvvm.read.ptx.sreg.ctaid.z");
    sreg!(block_dim_x, "llvm.nvvm.read.ptx.sreg.ntid.x");
    sreg!(block_dim_y, "llvm.nvvm.read.ptx.sreg.ntid.y");
    sreg!(block_dim_z, "llvm.nvvm.read.ptx.sreg.ntid.z");
    sreg!(grid_dim_x, "llvm.nvvm.read.ptx.sreg.nctaid.x");
    sreg!(grid_dim_y, "llvm.nvvm.read.ptx.sreg.nctaid.y");
    sreg!(grid_dim_z, "llvm.nvvm.read.ptx.sreg.nctaid.z");
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
