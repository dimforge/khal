/// Subgroup (wave/warp) floating-point add reduction.
///
/// Returns the sum of `val` across all invocations in the subgroup.
///
/// On SPIR-V: calls `spirv_std::arch::subgroup_f_add`.
/// On CUDA: uses warp-level `__shfl_xor_sync` butterfly reduction.
/// On CPU: returns `val` unchanged (subgroup size = 1).
#[inline(always)]
pub fn subgroup_f_add(val: f32) -> f32 {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::subgroup_f_add(val)
    }
    #[cfg(any(target_arch = "nvptx64", feature = "cuda-oxide"))]
    {
        warp_reduce_add(val)
    }
    #[cfg(not(any(
        target_arch = "spirv",
        target_arch = "nvptx64",
        feature = "cuda-oxide"
    )))]
    {
        val
    }
}

/// Subgroup (wave/warp) floating-point max reduction.
///
/// Returns the maximum of `val` across all invocations in the subgroup.
///
/// On SPIR-V: calls `spirv_std::arch::subgroup_f_max`.
/// On CUDA: uses warp-level `__shfl_xor_sync` butterfly reduction.
/// On CPU: returns `val` unchanged (subgroup size = 1).
#[inline(always)]
pub fn subgroup_f_max(val: f32) -> f32 {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::subgroup_f_max(val)
    }
    #[cfg(any(target_arch = "nvptx64", feature = "cuda-oxide"))]
    {
        warp_reduce_max(val)
    }
    #[cfg(not(any(
        target_arch = "spirv",
        target_arch = "nvptx64",
        feature = "cuda-oxide"
    )))]
    {
        val
    }
}

// cuda-oxide: warp shuffle through `cuda_device::warp`, recognized by name
// by the MIR importer on any compilation target.
#[cfg(feature = "cuda-oxide")]
#[inline(always)]
fn shfl_xor_sync(val: f32, lane_mask: u32) -> f32 {
    cuda_device::warp::shuffle_xor_f32(val, lane_mask)
}

// rust-cuda (rustc_codegen_nvvm): inline PTX.
#[cfg(all(target_arch = "nvptx64", not(feature = "cuda-oxide")))]
#[inline(always)]
fn shfl_xor_sync(val: f32, lane_mask: u32) -> f32 {
    let result: f32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.bfly.b32 {result}, {val}, {lane_mask}, 0x1f, 0xffffffff;",
            result = out(reg32) result,
            val = in(reg32) val,
            lane_mask = in(reg32) lane_mask,
        );
    }
    result
}

#[cfg(any(target_arch = "nvptx64", feature = "cuda-oxide"))]
#[inline(always)]
fn warp_reduce_add(mut val: f32) -> f32 {
    val += shfl_xor_sync(val, 16);
    val += shfl_xor_sync(val, 8);
    val += shfl_xor_sync(val, 4);
    val += shfl_xor_sync(val, 2);
    val += shfl_xor_sync(val, 1);
    val
}

#[cfg(any(target_arch = "nvptx64", feature = "cuda-oxide"))]
#[inline(always)]
fn warp_reduce_max(mut val: f32) -> f32 {
    let other = shfl_xor_sync(val, 16);
    if other > val {
        val = other;
    }
    let other = shfl_xor_sync(val, 8);
    if other > val {
        val = other;
    }
    let other = shfl_xor_sync(val, 4);
    if other > val {
        val = other;
    }
    let other = shfl_xor_sync(val, 2);
    if other > val {
        val = other;
    }
    let other = shfl_xor_sync(val, 1);
    if other > val {
        val = other;
    }
    val
}
