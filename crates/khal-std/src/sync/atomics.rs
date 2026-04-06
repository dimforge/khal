// =============================================================================
// Atomics — QueueFamily (storage buffer) scope
// =============================================================================

/// Atomically adds `value` to `*ptr` and returns the old value.
#[inline(always)]
pub fn atomic_add_i32(ptr: &mut i32, value: i32) -> i32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_i_add::<
            i32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicI32, Ordering};
        let atomic = unsafe { &*(ptr as *mut i32 as *const AtomicI32) };
        atomic.fetch_add(value, Ordering::Relaxed)
    }
}

/// Atomically adds `value` to `*ptr` (u32) and returns the old value.
#[inline(always)]
pub fn atomic_add_u32(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_i_add::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_add(value, Ordering::Relaxed)
    }
}

/// Atomically computes max(`*ptr`, `value`) and returns the old value.
#[inline(always)]
pub fn atomic_max_u32(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_u_max::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_max(value, Ordering::Relaxed)
    }
}

/// Atomically computes min(`*ptr`, `value`) and returns the old value.
#[inline(always)]
pub fn atomic_min_u32(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_u_min::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_min(value, Ordering::Relaxed)
    }
}

/// Atomically loads the value at `*ptr`.
///
/// Accepts `&mut u32` for compatibility with SPIR-V's pointer model where
/// all atomic operations require mutable pointers. Use `atomic_load_u32_shared`
/// when only a shared reference is available.
#[inline(always)]
pub fn atomic_load_u32(ptr: &mut u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_load::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.load(Ordering::Relaxed)
    }
}

/// Atomically loads the value at `*ptr` from a shared reference.
#[inline(always)]
pub fn atomic_load_u32_shared(ptr: &u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_load::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *const u32 as *const AtomicU32) };
        atomic.load(Ordering::Relaxed)
    }
}

/// Atomically exchanges `*ptr` with `value` and returns the old value.
#[inline(always)]
pub fn atomic_exchange_u32(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_exchange::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.swap(value, Ordering::Relaxed)
    }
}

/// Atomically compares `*ptr` with `comparator` and, if equal, replaces with `value`.
/// Returns the old value at `*ptr`.
#[inline(always)]
pub fn atomic_compare_exchange_u32(ptr: &mut u32, value: u32, comparator: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_compare_exchange::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value, comparator)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        match atomic.compare_exchange(comparator, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(old) => old,
            Err(old) => old,
        }
    }
}

/// Atomically ORs `*ptr` with `value` and returns the old value.
#[inline(always)]
pub fn atomic_or_u32(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_or::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_or(value, Ordering::Relaxed)
    }
}

/// Atomically ANDs `*ptr` with `value` and returns the old value.
#[inline(always)]
pub fn atomic_and_u32(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_and::<
            u32,
            { spirv_std::memory::Scope::QueueFamily as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_and(value, Ordering::Relaxed)
    }
}

/// Atomically adds a float `value` to the f32 stored (as u32 bits) at `ptr`.
///
/// Uses a compare-and-swap loop to emulate float atomic add:
/// repeatedly loads the current bits, reinterprets as f32, adds `value`,
/// and attempts to CAS the result back. The pointer must hold the bit
/// pattern of a valid f32 (initialize with `0u32` for 0.0).
#[inline]
pub fn atomic_add_f32(ptr: &mut u32, value: f32) {
    loop {
        let old_bits = atomic_load_u32(ptr);
        let new_val = f32::from_bits(old_bits) + value;
        let new_bits = new_val.to_bits();
        let prev = atomic_compare_exchange_u32(ptr, new_bits, old_bits);
        if prev == old_bits {
            break;
        }
    }
}

// =============================================================================
// Atomics — Workgroup scope (shared memory)
// =============================================================================

/// Atomically adds `value` to `*ptr` (workgroup scope) and returns the old value.
#[inline(always)]
pub fn atomic_add_u32_workgroup(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_i_add::<
            u32,
            { spirv_std::memory::Scope::Workgroup as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_add(value, Ordering::Relaxed)
    }
}

/// Atomically computes max(`*ptr`, `value`) (workgroup scope) and returns the old value.
#[inline(always)]
pub fn atomic_max_u32_workgroup(ptr: &mut u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_u_max::<
            u32,
            { spirv_std::memory::Scope::Workgroup as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.fetch_max(value, Ordering::Relaxed)
    }
}

/// Atomically stores `value` to `*ptr` (workgroup scope).
#[inline(always)]
pub fn atomic_store_u32_workgroup(ptr: &mut u32, value: u32) {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_exchange::<
            u32,
            { spirv_std::memory::Scope::Workgroup as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr, value);
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.store(value, Ordering::Relaxed);
    }
}

/// Atomically loads `*ptr` (workgroup scope).
#[inline(always)]
pub fn atomic_load_u32_workgroup(ptr: &mut u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        spirv_std::arch::atomic_load::<
            u32,
            { spirv_std::memory::Scope::Workgroup as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(ptr)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        use core::sync::atomic::{AtomicU32, Ordering};
        let atomic = unsafe { &*(ptr as *mut u32 as *const AtomicU32) };
        atomic.load(Ordering::Relaxed)
    }
}
