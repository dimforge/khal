# Changelog

_Disclaimer: this changelog is updated using generative AI, but is still verified manually._

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
