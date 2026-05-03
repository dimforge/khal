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
//!
//! Coroutine stacks are pooled per-thread to avoid repeated mmap/munmap
//! syscalls across dispatches.

extern crate std;

use std::cell::Cell;
use std::cell::RefCell;

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
// Intra-workgroup dispatch (using corosensei coroutines with stack pooling)
// =============================================================================

/// Stack size for coroutines. Shader functions use very little stack space
/// (local variables and small arrays), so 64KB is more than sufficient.
const COROUTINE_STACK_SIZE: usize = 64 * 1024;

thread_local! {
    /// Pointer to the active Yielder (null when not in coroutine mode).
    /// Each coroutine sets this before calling the work function.
    static COROUTINE_YIELDER: Cell<*mut corosensei::Yielder<(), ()>> = const { Cell::new(std::ptr::null_mut()) };

    /// Pool of reusable coroutine stacks. Stacks are allocated on first use
    /// and returned to the pool after each dispatch, avoiding repeated
    /// mmap/munmap syscalls.
    static STACK_POOL: RefCell<Vec<corosensei::stack::DefaultStack>> = RefCell::new(Vec::new());
}

/// Takes `count` stacks from the thread-local pool, allocating new ones if needed.
fn take_stacks(count: usize) -> Vec<corosensei::stack::DefaultStack> {
    STACK_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let reusable = count.min(pool.len());
        let drain_start = pool.len() - reusable;
        let mut stacks: Vec<corosensei::stack::DefaultStack> = pool.drain(drain_start..).collect();
        for _ in stacks.len()..count {
            stacks.push(
                corosensei::stack::DefaultStack::new(COROUTINE_STACK_SIZE)
                    .expect("failed to allocate coroutine stack"),
            );
        }
        stacks
    })
}

/// Returns stacks to the thread-local pool for reuse.
fn return_stacks(stacks: impl IntoIterator<Item = corosensei::stack::DefaultStack>) {
    STACK_POOL.with(|pool| {
        pool.borrow_mut().extend(stacks);
    });
}

/// Dispatches `num_threads` virtual threads using cooperative coroutines.
///
/// Each coroutine calls `f(thread_id)`. When a coroutine calls `barrier_wait()`,
/// it yields to the scheduler. After all coroutines have yielded (i.e., all
/// reached the barrier), the scheduler resumes them for the next phase.
///
/// This runs on a single OS thread with zero OS scheduling overhead.
/// Coroutine stacks are pooled to avoid repeated allocation.
pub fn dispatch_workgroup_threads(num_threads: usize, f: impl Fn(u32) + Sync) {
    use corosensei::{Coroutine, CoroutineResult};

    // SAFETY: We transmute the lifetime of `f` to 'static so we can move it
    // into Coroutine closures. This is safe because dispatch_workgroup_threads
    // blocks until all coroutines complete, so `f` outlives all of them.
    let f_ref: &'static (dyn Fn(u32) + Sync) =
        unsafe { core::mem::transmute(&f as &(dyn Fn(u32) + Sync)) };

    // Take stacks from the pool (reuses existing ones, allocates only if needed).
    let stacks = take_stacks(num_threads);

    // Create one coroutine per virtual thread, using pooled stacks.
    let mut coroutines: Vec<Option<Coroutine<(), (), (), corosensei::stack::DefaultStack>>> =
        stacks
            .into_iter()
            .enumerate()
            .map(|(tid, stack)| {
                Some(Coroutine::with_stack(stack, move |yielder, ()| {
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
    // Completed coroutines have their stacks recovered for pooling.
    let mut recovered_stacks = Vec::with_capacity(num_threads);
    loop {
        let mut all_done = true;
        for i in 0..coroutines.len() {
            let result = coroutines[i].as_mut().map(|c| c.resume(()));
            match result {
                Some(CoroutineResult::Yield(())) => {
                    all_done = false;
                }
                Some(CoroutineResult::Return(())) => {
                    recovered_stacks.push(coroutines[i].take().unwrap().into_stack());
                }
                None => {}
            }
        }
        if all_done {
            break;
        }
    }

    // Return stacks to the pool for reuse by future dispatches.
    return_stacks(recovered_stacks);
}
