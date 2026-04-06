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
    #[cfg(target_arch = "nvptx64")]
    {
        // Call the LLVM intrinsic directly instead of cuda_std::thread::sync_threads()
        // so that LLVM sees the `convergent` attribute during optimization passes.
        // Without this, LLVM tail-duplicates the barrier into both sides of divergent
        // branches (if/else), causing threads to hit different bar.sync instructions
        // and deadlocking the block.
        // This fixes kernels with barriers that were otherwise hanging when using
        // cuda_std::thread::sync_thread() instead.
        unsafe extern "C" {
            #[link_name = "llvm.nvvm.barrier0"]
            fn nvvm_barrier0();
        }
        unsafe {
            nvvm_barrier0();
        }
        //     cuda_std::thread::sync_threads();
    }

    #[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
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
