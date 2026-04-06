//! Derive proc-macros for `khal`.

extern crate proc_macro;

use proc_macro::TokenStream;

mod shader;
mod shader_args;
mod spirv_bindgen;

#[proc_macro_derive(Shader, attributes(shader, entry_point))]
pub fn derive_shader(item: TokenStream) -> TokenStream {
    shader::derive_shader(item)
}

#[proc_macro_derive(
    ShaderArgs,
    attributes(storage, uniform, push_constant, workgroup_size)
)]
pub fn derive_shader_args(item: TokenStream) -> TokenStream {
    shader_args::derive_shader_args(item)
}

#[proc_macro_attribute]
pub fn spirv_bindgen(attr: TokenStream, item: TokenStream) -> TokenStream {
    spirv_bindgen::spirv_bindgen(attr, item)
}
