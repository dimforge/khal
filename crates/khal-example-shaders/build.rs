// Re-exports this crate's source location to host crates that build the
// shaders. Combined with `links = "khal-example-shaders"` in Cargo.toml, the
// `cargo::metadata=manifest_dir=...` line below is delivered to any direct
// `[build-dependencies]` consumer as `DEP_KHAL_EXAMPLE_SHADERS_MANIFEST_DIR`.
// The host crate's `build.rs` reads that variable (via
// `KhalBuilder::from_dependency`) to find the shader sources, which works
// identically for in-workspace path deps and for crates fetched from
// crates.io — so a published host crate can rebuild its shaders on the
// consumer's machine without bundling this crate's source itself.
fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    println!("cargo::metadata=manifest_dir={manifest_dir}");
    println!("cargo:rerun-if-changed=build.rs");
}
