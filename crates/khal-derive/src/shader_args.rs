use darling::FromField;
use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DataStruct};

/// Parsed shader binding attribute data from a field.
///
/// Usage: `#[storage(set = 0, index = 1)]` or `#[uniform(set = 0, index = 1)]`
/// For storage bindings, you can explicitly specify read_only: `#[storage(set = 0, index = 1, read_only = false)]`
#[derive(FromField, Default)]
#[darling(default, attributes(storage, uniform))]
struct BindingFieldAttr {
    set: Option<u32>,
    index: Option<u32>,
    /// If Some, explicitly specifies whether this storage binding is read-only.
    /// If None, defaults to checking if the field type is `&mut GpuBuffer<T>` (mutable = read-write).
    read_only: Option<bool>,
}

/// Check if a field type is mutable (e.g., `&mut GpuBuffer<T>` or `GpuBufferSliceMut<T>`).
fn is_mutable_reference(ty: &syn::Type) -> bool {
    if let syn::Type::Reference(ref_type) = ty
        && ref_type.mutability.is_some()
    {
        return true;
    }
    if let syn::Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "GpuBufferSliceMut"
    {
        return true;
    }
    false
}

/// Check if a field has the `#[push_constant]` attribute.
fn has_push_constant_attr(field: &syn::Field) -> bool {
    field
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("push_constant"))
}

/// Determine the descriptor type from field attributes.
fn get_descriptor_type_from_field(field: &syn::Field) -> Option<&'static str> {
    field.attrs.iter().find_map(|attr| {
        if attr.path().is_ident("uniform") {
            Some("uniform")
        } else if attr.path().is_ident("storage") {
            Some("storage")
        } else {
            None
        }
    })
}

/// Parses a `#[workgroup_size(x, y, z)]` attribute from a struct.
///
/// Returns `[x, y, z]` workgroup dimensions, defaulting to 1 for unspecified dimensions.
/// Returns `None` if the attribute is not present.
fn parse_workgroup_size_attr(attrs: &[syn::Attribute]) -> Option<[u32; 3]> {
    for attr in attrs {
        if !attr.path().is_ident("workgroup_size") {
            continue;
        }

        let mut dims = [1u32, 1, 1];

        // Parse as a parenthesized list of literals: workgroup_size(64, 1, 1)
        if let Ok(args) = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::LitInt, syn::Token![,]>::parse_terminated,
        ) {
            let args: Vec<_> = args.into_iter().collect();
            if !args.is_empty() {
                dims[0] = args[0].base10_parse().unwrap_or(1);
            }
            if args.len() > 1 {
                dims[1] = args[1].base10_parse().unwrap_or(1);
            }
            if args.len() > 2 {
                dims[2] = args[2].base10_parse().unwrap_or(1);
            }
            return Some(dims);
        }
    }
    None
}

pub(crate) fn derive_shader_args(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    // Extract workgroup size from #[workgroup_size(x, y, z)] attribute if present
    let workgroup_size = parse_workgroup_size_attr(&input.attrs).unwrap_or([1, 1, 1]);
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];
    let wg_z = workgroup_size[2];

    // Extract generics from the input struct (excluding the Backend generic)
    let mut generics = input.generics.clone();

    // Add 'b lifetime to impl generics
    generics.params.insert(0, syn::parse_quote!('b));

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

    match &input.data {
        Data::Struct(DataStruct { fields, .. }) => {
            /*
             * Field attributes.
             */
            let mut field_writes = vec![];
            let mut binding_entries = vec![]; // (set, index, descriptor_type_tokens)
            let mut push_constant_fields = vec![];
            let mut push_constant_types = vec![];

            for field in fields.iter() {
                let ident = field
                    .ident
                    .as_ref()
                    .expect("unnamed fields not supported");

                // Check for #[push_constant] attribute
                if has_push_constant_attr(field) {
                    push_constant_fields.push(ident.clone());
                    push_constant_types.push(field.ty.clone());
                    continue;
                }

                // Parse #[storage(...)] or #[uniform(...)] attribute if present
                let binding_attr = match BindingFieldAttr::from_field(field) {
                    Ok(attr) => attr,
                    Err(e) => return e.write_errors().into(),
                };

                if let Some(binding_index) = binding_attr.index {
                    // Field has explicit binding info
                    let descriptor_set = binding_attr.set.unwrap_or(0);

                    // Determine if the field is read-only:
                    // 1. If explicitly specified via read_only attribute, use that
                    // 2. Otherwise, infer from type: &mut GpuBuffer<T> -> read-write, &GpuBuffer<T> -> read-only
                    let read_only = binding_attr
                        .read_only
                        .unwrap_or(!is_mutable_reference(&field.ty));

                    // Determine descriptor type from attribute name
                    let descriptor_type = match get_descriptor_type_from_field(field) {
                        Some("uniform") => quote! { khal::backend::DescriptorType::Uniform },
                        Some("storage") => quote! { khal::backend::DescriptorType::Storage { read_only: #read_only } },
                        None => {
                            return syn::Error::new_spanned(
                                field,
                                "Field must use either #[storage] or #[uniform] attribute",
                            )
                            .to_compile_error()
                            .into();
                        }
                        Some(other) => {
                            return syn::Error::new_spanned(
                                field,
                                format!("Invalid descriptor type '{}': expected 'uniform' or 'storage'", other),
                            )
                            .to_compile_error()
                            .into();
                        }
                    };

                    // Collect binding info for bind_group_layouts()
                    binding_entries.push((descriptor_set, binding_index, descriptor_type.clone()));

                    field_writes.push(quote! {
                        {
                            let binding = khal::backend::ShaderBinding {
                                space: #descriptor_set,
                                index: #binding_index,
                                descriptor_type: #descriptor_type,
                            };
                            self.#ident.write_arg(binding, dispatch)?;
                        }
                    });
                }
                // Fields without binding attribute are skipped
            }

            // Generate PUSH_CONSTANT_SIZE calculation
            let push_constant_size = if push_constant_fields.is_empty() {
                quote! { 0 }
            } else {
                // Sum of sizes of all push constant fields
                let sizes = push_constant_types.iter().map(|ty| {
                    quote! { ::core::mem::size_of::<#ty>() }
                });
                quote! { ( #( #sizes )+* ) as u32 }
            };

            // Generate push constant writing code
            let push_constant_write = if push_constant_fields.is_empty() {
                quote! {}
            } else {
                let field_writes_pc = push_constant_fields.iter().map(|ident| {
                    quote! {
                        __push_data.extend_from_slice(bytemuck::bytes_of(&self.#ident));
                    }
                });
                quote! {
                    #[cfg(feature = "push_constants")]
                    {
                        use khal::backend::Dispatch as _;
                        let mut __push_data: Vec<u8> = Vec::with_capacity(Self::PUSH_CONSTANT_SIZE as usize);
                        #( #field_writes_pc )*
                        dispatch.set_push_constants(&__push_data);
                    }
                }
            };

            // Generate bind_group_layouts() implementation
            // Group bindings by set and generate code to build BindGroupLayoutInfo
            let binding_inserts = binding_entries.iter().map(|(set, index, desc_type)| {
                quote! {
                    bindings.push(khal::backend::ShaderBinding {
                        space: #set,
                        index: #index,
                        descriptor_type: #desc_type,
                    });
                }
            });

            quote! {
                #[automatically_derived]
                impl #impl_generics khal::shader::ShaderArgs<'b> for #struct_identifier #ty_generics
                #where_clause
                {
                    const PUSH_CONSTANT_SIZE: u32 = #push_constant_size;

                    fn bind_group_layouts() -> khal::shader::BindGroupLayoutInfo {
                        let mut bindings: Vec<khal::backend::ShaderBinding> = Vec::new();
                        #( #binding_inserts )*

                        // Group by space (set)
                        let max_set = bindings.iter().map(|b| b.space).max().unwrap_or(0);
                        let mut groups = vec![Vec::new(); (max_set + 1) as usize];
                        for binding in bindings {
                            groups[binding.space as usize].push(binding);
                        }
                        // Sort each group by binding index
                        for group in &mut groups {
                            group.sort_by_key(|b| b.index);
                        }

                        khal::shader::BindGroupLayoutInfo { groups }
                    }

                    fn write_arg<'c>(&'b self, _binding: khal::backend::ShaderBinding, dispatch: &mut khal::backend::GpuDispatch<'c>) -> Result<(), khal::shader::ShaderArgsError>
                    where 'b: 'c {
                        use khal::shader::ShaderArgs;
                        #(
                            #field_writes
                        )*
                        #push_constant_write
                        Ok(())
                    }
                }

                #[automatically_derived]
                impl khal::shader::ShaderArgsType for #struct_identifier<'static> {
                    type For<'a> = #struct_identifier<'a>;
                    const WORKGROUP_SIZE: [u32; 3] = [#wg_x, #wg_y, #wg_z];
                }
            }
        }
        _ => unimplemented!(),
    }
        .into()
}
