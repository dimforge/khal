#![cfg_attr(target_arch = "spirv", no_std)]

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn add_assign(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f32],
) {
    let thread_id = invocation_id.x as usize;
    if thread_id < a.len() {
        a[thread_id] += b[thread_id];
    }
}
