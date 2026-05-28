//! Build-script helpers for shader crates.
//!
//! Host-only module (gated out of GPU-target builds). Intended to be called
//! from a shader crate's `build.rs` — see [`setup_shader_crate_build`].

/// Standard `build.rs` setup that every shader crate should run.
///
/// Does three things:
///
/// 1. Emits `cargo::metadata=manifest_dir=<path>` so that host crates
///    consuming this shader crate via `KhalBuilder::from_dependency`
///    discover the shader sources both in-workspace and from
///    `crates.io`-fetched copies.
/// 2. Declares the `target_arch_is_gpu` cfg via `cargo::rustc-check-cfg`
///    so `#[cfg(target_arch_is_gpu)]` / `#[cfg(not(target_arch_is_gpu))]`
///    don't trip the `unexpected_cfgs` lint.
/// 3. Sets `target_arch_is_gpu` when compiling for any GPU target
///    (SPIR-V, NVPTX). The host CPU build sees it unset.
///
/// Call from `build.rs`:
///
/// ```no_run
/// khal_std::build_script::setup_shader_crate_build();
/// ```
///
/// The shader crate must list `khal-std` as a `[build-dependencies]` entry
/// (in addition to its regular `[dependencies]` use).
pub fn setup_shader_crate_build() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    println!("cargo::metadata=manifest_dir={manifest_dir}");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo::rustc-check-cfg=cfg(target_arch_is_gpu)");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if matches!(target_arch.as_str(), "spirv" | "nvptx64") {
        println!("cargo::rustc-cfg=target_arch_is_gpu");
    }
}
