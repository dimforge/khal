# Changelog

_Disclaimer: this changelog is updated using generative AI, but is still verified manually._

## v0.4.0

### Added
- **cuda-oxide support** as an alternative to rust-cuda for building CUDA kernels. Enable the `cuda-oxide` feature on `khal-builder` and on `khal-std` (with `default-features = false`; the new default `rust-cuda` feature keeps the previous behavior). `#[spirv_bindgen]` emits a matching kernel entry. (#13)
- **Compute graphs**: `GpuBackend::begin_capture`/`end_capture` record submitted work into a `GpuGraph` that is replayed with `GpuGraph::launch`. CUDA only for now (`GpuBackend::supports_graphs`); other backends return the new `GpuBackendError::Unsupported`.
- `GpuBackend::auto`: picks CUDA on sm_120+ devices when compiled in, WebGPU otherwise. Overridable with `KHAL_BACKEND=cuda|webgpu|metal`.
- `GpuReadback` is now non-blocking on CUDA (async copy into pinned host memory polled with an event).
- `Cuda::compute_capability`.
- `khal-builder`: CUDA PTX is assembled into a cubin with `ptxas` when available (avoids driver/toolkit PTX version mismatches; opt out with `KHAL_CUDA_KEEP_PTX=1`; target arch from `KHAL_CUDA_ARCH` or auto-detected). `KHAL_SKIP_SPIRV=1` skips the SPIR-V build, and `CUDA_OXIDE_SHADERS_PTX_<CRATE>` embeds a prebuilt PTX/cubin.
- `atomic_sub_u32` in `khal-std`.
- Debugging env vars for the CUDA backend: `KHAL_CUDA_TRACE`, `KHAL_CUDA_PROFILE` (with `cuda::dump_kernel_profile`), `KHAL_CUDA_ALLOC_TRACE`, `KHAL_CUDA_GRAPH_DOT`.

### Changed
- Bumped `spirv-std` and `spirv-std-macros` to `0.10.0`; shaders must now be built with `cargo-gpu 0.10.0`.
- **Breaking (CUDA):** indirect dispatches no longer read the workgroup count back to the host. Kernels are launched with a fixed number of resident blocks (`KHAL_CUDA_PERSISTENT_BLOCKS`, default 16×SM count) that loop over the virtual workgroups. This changes the generated kernel ABI (slice bindings now receive an element count instead of a byte length, plus a trailing indirect-args parameter), so CUDA shaders must be rebuilt. `KHAL_CUDA_INDIRECT_SYNC=1` restores the old readback path.
- The CUDA backend now runs on its own stream instead of the legacy default stream (required for graph capture).
- `load_function` errors on CUDA name the missing entry point. (#10)
- Metal now matches WebGPU semantics: buffers are zero-initialized, out-of-bounds reads return zero and writes are skipped, workgroup memory is zero-initialized, loops are bounded, fast math is disabled, and MSL targets Metal 3.0.
- WebGPU: `slow_read_buffer` reuses pooled staging buffers. On wasm, consecutive compute passes are merged into one to cut per-pass overhead.
- WebGPU: dispatches exceeding the 65535 per-dimension workgroup limit panic with an explicit message.
- `GpuBackend::is_cuda` is available without the `cuda` feature.
- `cargo-cuda` disables MIR jump-threading, which could duplicate workgroup barriers and deadlock kernels.

### Fixed
- Metal: `write_buffer` while GPU work is in flight no longer races with running kernels; it is now ordered after previously submitted work.
- WebGPU SPIR-V passthrough modules now declare their entry points to wgpu, which otherwise failed to load them on Vulkan with `Unable to find entry point`.

## v0.3.0

### Changed
- Bumped `wgpu` and `naga` to `30`, `metal` to `0.33`, `syn` to `3`, and `darling` to `0.24`.
- `WebGpuBackendError` gained a `MapRange` variant, since wgpu 30 made `BufferSlice::get_mapped_range` fallible. The two readback paths that can't propagate an error (`GpuReadback::try_take`, `GpuTimestamps::try_take`) report no result instead of copying, matching how they already handle a failed map.
- The `push_constants` feature is now built on wgpu 30's renamed "immediates" API. No khal-level API change.

## v0.2.1

### Added
- **Non-blocking GPU→CPU readback.** A new `GpuReadback` type stages a buffer copy and returns immediately: `request`/`request_copy` start the transfer and `try_take` polls for completion (`is_idle` reports whether a readback is in flight). `GpuTimestamps` gained matching `request_read`/`try_take`/`is_idle` for non-blocking timestamp readback. Both are driven by a new `Backend::poll` maintenance method (a non-blocking counterpart to `synchronize`) implemented across the WebGPU, CUDA, Metal, and CPU backends. The `metal` feature now pulls in `block` for command-buffer completion handlers.
- `WebGpu::from_device`: build a WebGPU backend on top of an already-created wgpu instance/adapter/device/queue instead of creating its own.
- `from_wgpu` constructors on `GpuBufferSlice`, `GpuBufferSliceMut`, and `WebGpuBufferSlice` to wrap a foreign `wgpu::Buffer` (owned by another library sharing the device) as a khal buffer slice.
- `force_cpu_coroutines` option for `#[spirv_bindgen]`: forces the CPU backend to dispatch a kernel through the coroutine scheduler so `barrier_wait()` synchronizes, even for kernels with no `#[spirv(workgroup)]` parameter. Replaces the old dummy-workgroup-parameter hack.
- The CUDA backend's `load_module_bytes` now accepts a pre-linked CUBIN (detected via the ELF magic) in addition to PTX text, for modules referencing symbols the driver JIT cannot resolve on its own (e.g. libdevice `__nv_*` math).

### Fixed
- WebGPU entry-point lookup on wasm now mirrors naga's identifier sanitization, which appends a trailing `_` to names ending in a digit, so kernels like `reduce_add_f32` resolve correctly.

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
