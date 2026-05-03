use khal_builder::KhalBuilder;

fn main() {
    let output_dir = "shaders-spirv";

    // `from_dependency` resolves the shader crate's source directory by reading
    // the `DEP_KHAL_EXAMPLE_SHADERS_MANIFEST_DIR` env var that cargo
    // populates from the shader crate's `links` metadata (see its
    // `build.rs`). Prefer this over a hard-coded `"../khal-example-shaders"`
    // path: the relative path only resolves inside the workspace and
    // disappears from the published artifact, whereas the env var is set
    // for both workspace and registry-fetched copies of the shader crate.
    KhalBuilder::from_dependency("khal-example-shaders", true).build(output_dir);
}
