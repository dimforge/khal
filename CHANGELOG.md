# Changelog

_Disclaimer: this changelog is updated using generative AI, but is still verified manually._

## v0.2.0

### Added
- A native **Metal compute backend** (`metal` feature, macOS only). It translates SPIR-V to MSL with `naga` at function-load time and drives Apple's Metal API directly via the `metal` crate, rather than going through `wgpu`. Includes buffer management, indirect dispatch, push constants, and GPU timestamp queries. Compiled MSL libraries are cached by SPIR-V content hash. (#3)
- `GpuPass::memory_barrier` (and an `Encoder::memory_barrier` trait method): inserts a buffer-scope memory barrier between dispatches within a compute pass. Required on Metal — which uses `MTLDispatchType::Concurrent` and does not auto-synchronize consecutive dispatches — and a no-op on backends that already insert implicit barriers (WebGPU, CUDA, CPU). (#3)
- `khal_std::build_script::setup_shader_crate_build()`: a `build.rs` helper for shader crates that emits the `manifest_dir` metadata used by `KhalBuilder::from_dependency`, and declares/sets the `target_arch_is_gpu` cfg (set for SPIR-V/NVPTX targets, unset on host CPU builds). (#3)
- `GpuBackend::is_metal` and `Backend::as_metal` accessors. (#3)

### Changed
- The WebGPU backend now compiles shader modules with `force_loop_bounding: true` (instead of fully unchecked) to work around an apparent miscompilation of loops on some platforms (Windows + Nvidia). (#3)
- Bumped `glamx` from `0.2` to `0.3` in `khal-std`, enabling its `u32`, `i32`, and `f64` features. (#3)

## v0.1.1

### Added
- `KhalBuilder::from_dependency` (in `khal-builder`): locates the shader crate via cargo's `links` metadata mechanism instead of a hard-coded relative path. This lets a published host crate rebuild its shaders on the consumer's machine using a registry-fetched copy of the shader crate, without needing to bundle the shader sources in the host's published artifact.

### Changed
- The `khal-example` tutorial crate now uses `KhalBuilder::from_dependency` instead of a hard-coded `"../khal-example-shaders"` path, and `khal-example-shaders` declares `links = "khal-example-shaders"` plus a small `build.rs` that re-exports its `CARGO_MANIFEST_DIR` to dependents. This is the recommended pattern for downstream crates that publish to crates.io.

## v0.1.0

This shows the changes between the time of open-sourcing the crate and its first release to crates.io:

### Added
- `println!` support for shaders running on the CPU backend (`khal-std`).

### Changed
- Switch `spirv-std` and `spirv-std-macros` to the published `0.10.0-alpha.1` release (previously pinned to a git revision).
- Cache coroutines on the CPU backend for improved performance.
- Enable incremental builds in the workspace to work around a `rust-gpu` issue where the example shader entrypoint was being dropped.
