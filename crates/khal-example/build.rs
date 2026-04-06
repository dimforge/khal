use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../khal-example-shaders";
    let output_dir = "shaders-spirv";

    KhalBuilder::new(shader_crate, true).build(output_dir);
}
