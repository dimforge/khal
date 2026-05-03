# Changelog

_Disclaimer: this changelog is updated using generative AI, but is still verified manually._

## v0.1.0

This shows the changes between the time of open-sourcing the crate and its first release to crates.io:

### Added
- `println!` support for shaders running on the CPU backend (`khal-std`).

### Changed
- Switch `spirv-std` and `spirv-std-macros` to the published `0.10.0-alpha.1` release (previously pinned to a git revision).
- Cache coroutines on the CPU backend for improved performance.
- Enable incremental builds in the workspace to work around a `rust-gpu` issue where the example shader entrypoint was being dropped.
