#![cfg_attr(target_arch = "spirv", no_std)]

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn add_assign(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f32],
) {
    let thread_id = invocation_id.x as usize;
    if thread_id < a.len() && thread_id < b.len() {
        *a.at_mut(thread_id) += b.read(thread_id);
    }
}

/// 2D-grid probe: writes a tag encoding (batch=y, x, src[y]) per slot.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn grid_probe(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] src: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [u32],
) {
    let x = invocation_id.x;
    let y = invocation_id.y;
    let cap = 64u32;
    if x >= cap || y as usize >= src.len() {
        return;
    }
    out.write((y * cap + x) as usize, (y << 20) | (src.read(y as usize) << 10) | x);
}

/// Batched StepRng probe mimicking the nexus narrow-phase pattern:
/// per-batch len + strided loop + batch-sliced reads.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn batch_probe(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lens: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [u32],
) {
    let num_threads = num_workgroups.x * 64;
    let batch = invocation_id.y;
    if batch as usize >= lens.len() {
        return;
    }
    let cap = 64u32;
    let len = lens.read(batch as usize).min(cap);
    for i in khal_std::iter::StepRng::new(invocation_id.x..len, num_threads) {
        let v = src.read((batch * cap + i) as usize);
        out.write((batch * cap + i) as usize, (batch << 20) | v);
    }
}
