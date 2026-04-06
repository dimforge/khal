use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DataStruct};

fn default_spirv_path() -> String {
    "crate::SPIRV_DIR".to_string()
}

#[derive(FromDeriveInput, Clone)]
#[darling(attributes(shader), default)]
struct DeriveShadersParams {
    /// Path expression (as a string) to a `Dir<'static>` constant containing embedded SPIR-V files.
    /// Defaults to `"crate::SPIRV_DIR"`.
    #[darling(default = "default_spirv_path")]
    pub spirv: String,
}

impl Default for DeriveShadersParams {
    fn default() -> Self {
        Self {
            spirv: default_spirv_path(),
        }
    }
}

/// Extracts a string value from a `#[name = "value"]` attribute on a field.
fn parse_field_string_attr(attrs: &[syn::Attribute], name: &str) -> Option<String> {
    attrs.iter().find_map(|attr| {
        if attr.path().is_ident(name)
            && let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(lit) = &nv.value
            && let syn::Lit::Str(s) = &lit.lit
        {
            return Some(s.value());
        }
        None
    })
}

pub(crate) fn derive_shader(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    let derive_shaders = match DeriveShadersParams::from_derive_input(&input) {
        Ok(v) => v,
        Err(e) => {
            return e.write_errors().into();
        }
    };

    match &input.data {
        Data::Struct(DataStruct { fields, .. }) => {
            generate_comptime_impl(&input, struct_identifier, &derive_shaders, fields)
        }
        _ => unimplemented!(),
    }
    .into()
}

fn generate_comptime_impl(
    _input: &syn::DeriveInput,
    struct_identifier: &syn::Ident,
    derive_shaders: &DeriveShadersParams,
    fields: &syn::Fields,
) -> proc_macro2::TokenStream {
    let spirv_dir_path: syn::Expr =
        syn::parse_str(&derive_shaders.spirv).expect("Invalid path expression in spirv attribute");

    let mut field_initializers = vec![];
    let mut feature_checks = vec![];

    for field in fields.iter() {
        let field_ident = field.ident.as_ref().expect("unnamed fields not supported");
        let field_type = &field.ty;

        // Parse optional field-level #[entry_point = "..."] attribute.
        let entry_point_attr = parse_field_string_attr(&field.attrs, "entry_point");

        let init = if let Some(entry) = entry_point_attr {
            // spirv_bindgen wrapper type with overridden entry point.
            quote! {
                #field_ident: <#field_type>::from_dir_with_entry_point(backend, &#spirv_dir_path, #entry)?
            }
        } else {
            // spirv_bindgen wrapper type with default entry point.
            quote! {
                #field_ident: <#field_type>::from_dir(backend, &#spirv_dir_path)?
            }
        };

        field_initializers.push(init);

        // Verify that backend features are propagated to the shader crate.
        // If this crate enables e.g. `cpu` but the shader crate doesn't, referencing
        // the marker constant will fail to compile, catching the misconfiguration early.
        feature_checks.push(quote! {
            #[cfg(feature = "cpu")]
            const _: () = <#field_type>::__ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU;
            #[cfg(feature = "cpu-parallel")]
            const _: () = <#field_type>::__ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU_PARALLEL;
            #[cfg(feature = "cuda")]
            const _: () = <#field_type>::__ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CUDA;
        });
    }

    quote! {
        // Compile-time checks that backend features are properly propagated
        // to the shader crates referenced by this struct's fields.
        #(#feature_checks)*

        #[automatically_derived]
        impl khal::shader::Shader for #struct_identifier {
            fn from_backend(backend: &khal::backend::GpuBackend) -> Result<Self, khal::backend::GpuBackendError> {
                Ok(Self {
                    #(
                        #field_initializers,
                    )*
                })
            }
        }
    }
}
