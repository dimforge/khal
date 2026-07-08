/// Workgroup memory barrier with group synchronization.
///
/// On GPU: calls `spirv_std::arch::workgroup_memory_barrier_with_group_sync`.
/// On CPU: waits on the thread-local workgroup barrier (set by CPU dispatch).
#[inline(always)]
pub fn workgroup_memory_barrier_with_group_sync() {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    }
    // cuda-oxide: `cuda_device::sync_threads` is recognized by name by the
    // MIR importer on any compilation target and lowers to the convergent
    // NVVM barrier op (`bar.sync`), which optimization passes will not
    // tail-duplicate into divergent branches.
    #[cfg(all(feature = "cuda-oxide", not(target_arch = "spirv")))]
    {
        cuda_device::sync_threads();
    }
    #[cfg(all(target_arch = "nvptx64", not(feature = "cuda-oxide")))]
    {
        cuda_std::thread::sync_threads();
    }

    #[cfg(not(any(
        target_arch = "spirv",
        target_arch = "nvptx64",
        feature = "cuda-oxide"
    )))]
    #[cfg(feature = "cpu")]
    {
        crate::arch::cpu::barrier_wait();
    }
}

/// Control barrier with explicit execution scope, memory scope, and semantics.
///
/// On GPU (SPIR-V): calls `spirv_std::arch::control_barrier`.
/// On GPU (CUDA): calls `__syncthreads()`.
/// On CPU: waits on the thread-local workgroup barrier.
#[inline(always)]
pub fn control_barrier<const EXECUTION: u32, const MEMORY: u32, const SEMANTICS: u32>() {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::control_barrier::<EXECUTION, MEMORY, SEMANTICS>();
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        // handle CUDA and CPU backends
        workgroup_memory_barrier_with_group_sync();
    }
}
