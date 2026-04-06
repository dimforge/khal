//! CUDA intrinsic wrappers for thread and block indexing.

use cuda_std::thread;
use glamx::UVec3;

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
        cuda_std::thread::grid_dim_x(),
        cuda_std::thread::grid_dim_y(),
        cuda_std::thread::grid_dim_z(),
    )
}
