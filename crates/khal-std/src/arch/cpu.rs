//! CPU runtime support for GPU shader execution.
//!
//! Provides:
//! - Coroutine-based cooperative scheduling for intra-workgroup parallelism
//! - Rayon-based parallel dispatch across workgroups
//!
//! Shared-memory GPU kernels require barrier synchronization between threads
//! in a workgroup. Instead of using OS threads (expensive context switches),
//! we simulate GPU threads as lightweight stackful coroutines (via `corosensei`)
//! that yield at each barrier. A single OS thread runs all coroutines
//! cooperatively, with zero OS scheduling overhead.

extern crate std;

use std::cell::Cell;

// =============================================================================
// Barrier: yields the current coroutine back to the scheduler
// =============================================================================

/// Waits at the workgroup barrier.
///
/// In coroutine mode: yields the current coroutine to the scheduler. The
/// scheduler resumes all coroutines after every one has yielded at this barrier.
/// Outside coroutine mode: no-op (for non-shared-memory kernels).
pub fn barrier_wait() {
    COROUTINE_YIELDER.with(|cell| {
        let ptr = cell.get();
        if !ptr.is_null() {
            // SAFETY: The pointer is valid for the duration of the coroutine's
            // execution. It points to the Yielder on the coroutine's own stack
            // inside `dispatch_workgroup_threads`.
            unsafe { (*ptr).suspend(()) };
        }
    });
}

/// Legacy API (no-op, kept for backward compatibility with the coroutine scheduler).
pub fn set_barrier(_barrier: std::sync::Arc<std::sync::Barrier>) {}
/// Legacy API (no-op, kept for backward compatibility with the coroutine scheduler).
pub fn clear_barrier() {}

// =============================================================================
// Cross-workgroup dispatch
// =============================================================================

/// Dispatches `num_workgroups` tasks.
///
/// With the `parallel` feature: uses rayon for parallel execution.
/// Without: runs workgroups sequentially.
pub fn dispatch_workgroups(num_workgroups: usize, f: impl Fn(u32) + Sync + Send) {
    #[cfg(feature = "cpu-parallel")]
    {
        use rayon::prelude::*;
        (0..num_workgroups as u32).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "cpu-parallel"))]
    {
        for i in 0..num_workgroups as u32 {
            f(i);
        }
    }
}

// =============================================================================
// Intra-workgroup dispatch (using corosensei coroutines)
// =============================================================================

thread_local! {
    /// Pointer to the active Yielder (null when not in coroutine mode).
    /// Each coroutine sets this before calling the work function.
    static COROUTINE_YIELDER: Cell<*mut corosensei::Yielder<(), ()>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Dispatches `num_threads` virtual threads using cooperative coroutines.
///
/// Each coroutine calls `f(thread_id)`. When a coroutine calls `barrier_wait()`,
/// it yields to the scheduler. After all coroutines have yielded (i.e., all
/// reached the barrier), the scheduler resumes them for the next phase.
///
/// This runs on a single OS thread with zero OS scheduling overhead.
pub fn dispatch_workgroup_threads(num_threads: usize, f: impl Fn(u32) + Sync) {
    use corosensei::{Coroutine, CoroutineResult};

    // SAFETY: We transmute the lifetime of `f` to 'static so we can move it
    // into Coroutine closures. This is safe because dispatch_workgroup_threads
    // blocks until all coroutines complete, so `f` outlives all of them.
    let f_ref: &'static (dyn Fn(u32) + Sync) =
        unsafe { core::mem::transmute(&f as &(dyn Fn(u32) + Sync)) };

    // Create one coroutine per virtual thread.
    let mut coroutines: Vec<Option<Coroutine<(), (), ()>>> = (0..num_threads)
        .map(|tid| {
            Some(Coroutine::new(move |yielder, ()| {
                // Store the yielder pointer in TLS so barrier_wait() can find it.
                COROUTINE_YIELDER.with(|cell| {
                    cell.set(yielder as *const _ as *mut _);
                });
                f_ref(tid as u32);
                // Clear the yielder pointer.
                COROUTINE_YIELDER.with(|cell| cell.set(std::ptr::null_mut()));
            }))
        })
        .collect();

    // Run all coroutines in round-robin until all complete.
    // Each "round" corresponds to one barrier synchronization point.
    loop {
        let mut all_done = true;
        for slot in coroutines.iter_mut() {
            if let Some(coroutine) = slot {
                match coroutine.resume(()) {
                    CoroutineResult::Yield(()) => {
                        // Coroutine yielded at a barrier — continue to next one.
                        all_done = false;
                    }
                    CoroutineResult::Return(()) => {
                        // Coroutine completed — remove it.
                        *slot = None;
                    }
                }
            }
        }
        if all_done {
            break;
        }
    }
}
