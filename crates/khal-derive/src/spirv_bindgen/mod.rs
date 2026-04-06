use proc_macro::TokenStream;
use quote::{ToTokens, quote};

mod cpu;
mod cuda;

// ── Types ────────────────────────────────────────────────────────────────────

/// Parsed information about a shader kernel parameter binding.
pub(super) struct ShaderBinding {
    /// Parameter name
    pub name: syn::Ident,
    /// Descriptor set (default 0)
    pub descriptor_set: u32,
    /// Binding index within the set
    pub binding: u32,
    /// Whether this is a uniform (true) or storage buffer (false)
    pub is_uniform: bool,
    /// Whether the binding is mutable (&mut vs &)
    pub is_mutable: bool,
    /// The element type (e.g., `u32` from `&[u32]` or `&u32`)
    pub element_type: syn::Type,
    /// Optional cfg attributes that apply to this binding
    pub cfg_attrs: Vec<syn::Attribute>,
}

/// Parsed information about a push constant parameter.
pub(super) struct PushConstantBinding {
    /// Parameter name
    pub name: syn::Ident,
    /// The type of the push constant
    pub ty: syn::Type,
    /// Optional cfg attributes that apply to this binding
    pub cfg_attrs: Vec<syn::Attribute>,
}

/// The specific kind of a spirv built-in parameter.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum BuiltinKind {
    GlobalInvocationId,
    LocalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    LocalInvocationIndex,
    SubgroupId,
    SubgroupLocalInvocationId,
    Other,
}

/// Result of parsing a spirv attribute.
enum SpirvAttrKind {
    /// A storage or uniform buffer binding: (is_uniform, descriptor_set, binding_index)
    Binding(bool, u32, u32),
    /// A push constant parameter
    PushConstant,
    /// A built-in parameter with specific kind
    Builtin(BuiltinKind),
    /// A workgroup (shared memory) parameter
    Workgroup,
}

/// Tracks a function parameter in its original declaration order for CPU dispatch.
pub(super) struct OriginalParam {
    pub name: syn::Ident,
    pub kind: OriginalParamKind,
    /// The full parameter type from the original function signature.
    pub ty: syn::Type,
    pub cfg_attrs: Vec<syn::Attribute>,
}

pub(super) enum OriginalParamKind {
    Builtin(BuiltinKind),
    Binding { is_uniform: bool, is_mutable: bool },
    PushConstant,
    Workgroup,
}

/// Result of extracting type information from a reference.
struct ExtractedType {
    /// The element type (e.g., `u32` from `&[u32]`)
    element_type: syn::Type,
    /// Whether the reference is mutable
    is_mutable: bool,
}

// ── Utility functions ────────────────────────────────────────────────────────

/// Parses a spirv attribute to extract binding information.
///
/// Handles attributes like:
/// - `#[spirv(storage_buffer, descriptor_set = 0, binding = 0)]`
/// - `#[spirv(uniform, descriptor_set = 0, binding = 1)]`
/// - `#[spirv(push_constant)]`
/// - `#[spirv(global_invocation_id)]` (returns Builtin)
fn parse_spirv_attr(attr: &syn::Attribute) -> Option<SpirvAttrKind> {
    // Check if this is a spirv attribute
    if !attr.path().is_ident("spirv") {
        return None;
    }

    let mut is_uniform = false;
    let mut is_storage = false;
    let mut is_push_constant = false;
    let mut builtin_kind: Option<BuiltinKind> = None;
    let mut is_workgroup = false;
    let mut descriptor_set: Option<u32> = None;
    let mut binding: Option<u32> = None;

    // Parse the attribute arguments
    let _ = attr.parse_nested_meta(|meta| {
        let ident_str = meta.path.get_ident().map(|i| i.to_string());

        match ident_str.as_deref() {
            Some("uniform") => {
                is_uniform = true;
            }
            Some("storage_buffer") => {
                is_storage = true;
            }
            Some("push_constant") => {
                is_push_constant = true;
            }
            Some("descriptor_set") => {
                let value: syn::LitInt = meta.value()?.parse()?;
                descriptor_set = Some(value.base10_parse()?);
            }
            Some("binding") => {
                let value: syn::LitInt = meta.value()?.parse()?;
                binding = Some(value.base10_parse()?);
            }
            Some("global_invocation_id") => {
                builtin_kind = Some(BuiltinKind::GlobalInvocationId);
            }
            Some("local_invocation_id") => {
                builtin_kind = Some(BuiltinKind::LocalInvocationId);
            }
            Some("workgroup_id") => {
                builtin_kind = Some(BuiltinKind::WorkgroupId);
            }
            Some("num_workgroups") => {
                builtin_kind = Some(BuiltinKind::NumWorkgroups);
            }
            Some("local_invocation_index") => {
                builtin_kind = Some(BuiltinKind::LocalInvocationIndex);
            }
            Some("subgroup_id") => {
                builtin_kind = Some(BuiltinKind::SubgroupId);
            }
            Some("subgroup_local_invocation_id") => {
                builtin_kind = Some(BuiltinKind::SubgroupLocalInvocationId);
            }
            Some("workgroup") => {
                is_workgroup = true;
            }
            Some("vertex_index") | Some("instance_index") | Some("position") => {
                builtin_kind = Some(BuiltinKind::Other);
            }
            Some("compute") => {
                // Skip compute(threads(...)) - it's on the function, not parameters
            }
            _ => {
                // Unknown attribute part, skip
            }
        }
        Ok(())
    });

    // Return appropriate kind
    if is_push_constant {
        Some(SpirvAttrKind::PushConstant)
    } else if (is_uniform || is_storage) && binding.is_some() {
        Some(SpirvAttrKind::Binding(
            is_uniform,
            descriptor_set.unwrap_or(0),
            binding.unwrap(),
        ))
    } else if is_workgroup {
        Some(SpirvAttrKind::Workgroup)
    } else {
        builtin_kind.map(SpirvAttrKind::Builtin)
    }
}

/// Extracts cfg attributes from a list of attributes.
fn extract_cfg_attrs(attrs: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr"))
        .cloned()
        .collect()
}

/// Extracts the element type from a reference type.
///
/// - `&[T]` -> `T`, is_slice = true
/// - `&mut [T]` -> `T`, is_slice = true
/// - `&T` -> `T`, is_slice = false
fn extract_element_type(ty: &syn::Type) -> Option<ExtractedType> {
    if let syn::Type::Reference(ref_type) = ty {
        let is_mutable = ref_type.mutability.is_some();
        let inner = &*ref_type.elem;

        // For slices (&[T] / &mut [T]), unwrap to get the element type T.
        // For everything else (&T, &mut T, &[T; N]), keep the inner type as-is.
        let element_type = if let syn::Type::Slice(slice) = inner {
            *slice.elem.clone()
        } else {
            inner.clone()
        };

        return Some(ExtractedType {
            element_type,
            is_mutable,
        });
    }
    None
}

/// Parses the workgroup size from a `#[spirv(compute(threads(...)))]` attribute.
///
/// Returns `[x, y, z]` workgroup dimensions, defaulting to 1 for unspecified dimensions.
/// Returns `None` if the attribute is not a compute shader attribute.
fn parse_workgroup_size(attr: &syn::Attribute) -> Option<[u32; 3]> {
    if !attr.path().is_ident("spirv") {
        return None;
    }

    let mut workgroup_size: Option<[u32; 3]> = None;

    let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("compute") {
            // Parse compute(threads(...))
            meta.parse_nested_meta(|inner| {
                if inner.path.is_ident("threads") {
                    // Parse threads(x) or threads(x, y) or threads(x, y, z)
                    let content;
                    syn::parenthesized!(content in inner.input);

                    let mut dims = [1u32, 1, 1];

                    // Parse first dimension (required)
                    let x: syn::LitInt = content.parse()?;
                    dims[0] = x.base10_parse()?;

                    // Parse optional second dimension
                    if content.peek(syn::Token![,]) {
                        let _: syn::Token![,] = content.parse()?;
                        let y: syn::LitInt = content.parse()?;
                        dims[1] = y.base10_parse()?;

                        // Parse optional third dimension
                        if content.peek(syn::Token![,]) {
                            let _: syn::Token![,] = content.parse()?;
                            let z: syn::LitInt = content.parse()?;
                            dims[2] = z.base10_parse()?;
                        }
                    }

                    workgroup_size = Some(dims);
                }
                Ok(())
            })?;
        }
        Ok(())
    });

    workgroup_size
}

/// Converts a snake_case identifier to PascalCase.
fn snake_to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

/// Checks if a type is a reference to a slice (`&[T]` or `&mut [T]`).
pub(super) fn is_slice_reference(ty: &syn::Type) -> bool {
    if let syn::Type::Reference(ref_type) = ty {
        matches!(&*ref_type.elem, syn::Type::Slice(_))
    } else {
        false
    }
}

// ── Main implementation ──────────────────────────────────────────────────────

/// Generates a ShaderArgs struct from a GPU kernel function signature.
///
/// # Usage
///
/// The struct name is optional. If not provided, it is derived from the function name
/// by converting it from snake_case to PascalCase.
///
/// ```ignore
/// #[spirv(compute(threads(64)))]
/// #[spirv_bindgen]  // Generates `MyKernel` struct
/// pub fn my_kernel(
///     #[spirv(global_invocation_id)] invocation_id: UVec3,
///     #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &[u32],
///     #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
///     #[spirv(uniform, descriptor_set = 0, binding = 2)] config: &u32,
/// ) { ... }
/// ```
///
/// You can also specify an explicit struct name:
///
/// ```ignore
/// #[spirv(compute(threads(64)))]
/// #[spirv_bindgen(CustomName)]
/// pub fn my_kernel(...) { ... }
/// ```
///
/// This generates (on non-spirv targets):
///
/// ```ignore
/// #[derive(khal::ShaderArgs)]
/// pub struct MyKernel<'a> {
///     #[storage(set = 0, index = 0)]
///     pub data: &'a khal::backend::GpuBuffer<u32>,
///     #[storage(set = 0, index = 1)]
///     pub output: &'a mut khal::backend::GpuBuffer<u32>,
///     #[uniform(set = 0, index = 2)]
///     pub config: &'a khal::backend::GpuBuffer<u32>,
/// }
///
/// impl MyKernel<'_> {
///     pub const WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];
/// }
/// ```
pub(crate) fn spirv_bindgen(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the function first so we can derive struct name if needed
    let func = syn::parse_macro_input!(item as syn::ItemFn);

    // Parse optional struct name and flags from attribute.
    // Supported forms:
    //   #[spirv_bindgen]
    //   #[spirv_bindgen(MyName)]
    //   #[spirv_bindgen(spirv_passthrough)]
    //   #[spirv_bindgen(MyName, spirv_passthrough)]
    let mut spirv_passthrough = false;
    let struct_name: syn::Ident = if attr.is_empty() {
        // No explicit name provided - derive from function name (snake_case to PascalCase)
        let func_name = func.sig.ident.to_string();
        let pascal_name = snake_to_pascal_case(&func_name);
        syn::Ident::new(&pascal_name, func.sig.ident.span())
    } else {
        // Parse comma-separated identifiers
        let args = syn::parse_macro_input!(attr with syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated);

        let mut name = None;
        for ident in &args {
            if ident == "spirv_passthrough" {
                spirv_passthrough = true;
            } else {
                if name.is_some() {
                    return syn::Error::new_spanned(
                        ident,
                        "Multiple struct names specified in #[spirv_bindgen] attribute",
                    )
                    .to_compile_error()
                    .into();
                }
                name = Some(ident.clone());
            }
        }

        name.unwrap_or_else(|| {
            let func_name = func.sig.ident.to_string();
            let pascal_name = snake_to_pascal_case(&func_name);
            syn::Ident::new(&pascal_name, func.sig.ident.span())
        })
    };

    // Extract workgroup size from function attributes
    let workgroup_size = func
        .attrs
        .iter()
        .find_map(parse_workgroup_size)
        .unwrap_or([1, 1, 1]);

    // Extract bindings and push constants from function parameters
    let mut bindings: Vec<ShaderBinding> = vec![];
    let mut push_constants: Vec<PushConstantBinding> = vec![];
    // Track all parameters in original order for CPU dispatch generation.
    let mut original_params: Vec<OriginalParam> = vec![];

    for param in &func.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = param {
            // Get parameter name
            let param_name = if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                pat_ident.ident.clone()
            } else {
                continue;
            };

            // Extract cfg attributes from this parameter
            let cfg_attrs = extract_cfg_attrs(&pat_type.attrs);

            // Look for spirv attribute
            for attr in &pat_type.attrs {
                if let Some(kind) = parse_spirv_attr(attr) {
                    match kind {
                        SpirvAttrKind::Binding(is_uniform, descriptor_set, binding_index) => {
                            // Extract element type and mutability
                            if let Some(extracted) = extract_element_type(&pat_type.ty) {
                                bindings.push(ShaderBinding {
                                    name: param_name.clone(),
                                    descriptor_set,
                                    binding: binding_index,
                                    is_uniform,
                                    is_mutable: extracted.is_mutable,
                                    element_type: extracted.element_type,
                                    cfg_attrs: cfg_attrs.clone(),
                                });
                                original_params.push(OriginalParam {
                                    name: param_name.clone(),
                                    kind: OriginalParamKind::Binding {
                                        is_uniform,
                                        is_mutable: extracted.is_mutable,
                                    },
                                    ty: (*pat_type.ty).clone(),
                                    cfg_attrs: cfg_attrs.clone(),
                                });
                            }
                        }
                        SpirvAttrKind::PushConstant => {
                            // Extract the inner type from the reference
                            if let Some(extracted) = extract_element_type(&pat_type.ty) {
                                push_constants.push(PushConstantBinding {
                                    name: param_name.clone(),
                                    ty: extracted.element_type,
                                    cfg_attrs: cfg_attrs.clone(),
                                });
                                original_params.push(OriginalParam {
                                    name: param_name.clone(),
                                    kind: OriginalParamKind::PushConstant,
                                    ty: (*pat_type.ty).clone(),
                                    cfg_attrs: cfg_attrs.clone(),
                                });
                            }
                        }
                        SpirvAttrKind::Builtin(builtin_kind) => {
                            original_params.push(OriginalParam {
                                name: param_name.clone(),
                                kind: OriginalParamKind::Builtin(builtin_kind),
                                ty: (*pat_type.ty).clone(),
                                cfg_attrs: cfg_attrs.clone(),
                            });
                        }
                        SpirvAttrKind::Workgroup => {
                            original_params.push(OriginalParam {
                                name: param_name.clone(),
                                kind: OriginalParamKind::Workgroup,
                                ty: (*pat_type.ty).clone(),
                                cfg_attrs: cfg_attrs.clone(),
                            });
                        }
                    }
                    break;
                }
            }
        }
    }

    // Sort bindings by (descriptor_set, binding) for consistent output
    bindings.sort_by_key(|b| (b.descriptor_set, b.binding));

    // Generate struct fields for buffer bindings
    let binding_fields: Vec<proc_macro2::TokenStream> = bindings
        .iter()
        .map(|b| {
            let name = &b.name;
            let set = b.descriptor_set;
            let index = b.binding;
            let elem_ty = &b.element_type;
            let cfg_attrs = &b.cfg_attrs;

            // Always use GpuBufferSlice/GpuBufferSliceMut for all buffer bindings
            // (both slice types like &[T] and single-value references like &T)
            let attr = if b.is_uniform {
                quote! { #[uniform(set = #set, index = #index)] }
            } else {
                quote! { #[storage(set = #set, index = #index)] }
            };
            if b.is_mutable {
                quote! {
                    #(#cfg_attrs)*
                    #attr
                    pub #name: khal::backend::GpuBufferSliceMut<'a, #elem_ty>,
                }
            } else {
                quote! {
                    #(#cfg_attrs)*
                    #attr
                    pub #name: khal::backend::GpuBufferSlice<'a, #elem_ty>,
                }
            }
        })
        .collect();

    // Generate struct fields for push constants
    let push_constant_fields: Vec<proc_macro2::TokenStream> = push_constants
        .iter()
        .map(|pc| {
            let name = &pc.name;
            let ty = &pc.ty;
            let cfg_attrs = &pc.cfg_attrs;

            quote! {
                #(#cfg_attrs)*
                #[push_constant]
                pub #name: #ty,
            }
        })
        .collect();

    // Generate workgroup size constant
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];
    let wg_z = workgroup_size[2];

    // The args struct gets an "Args" suffix, the wrapper struct keeps the original name.
    let args_struct_name = syn::Ident::new(&format!("{}Args", struct_name), struct_name.span());

    // Generate the internal args struct definition (only on non-spirv targets)
    // The workgroup_size attribute is used by #[derive(ShaderArgs)] to generate
    // the WORKGROUP_SIZE constant in the ShaderArgsType impl.
    let args_doc = format!(
        "Arguments the [`{}`] GPU kernel build and pass to its internal `GpuFunction`.",
        struct_name
    );
    let args_struct_def = quote! {
        #[doc = #args_doc]
        #[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
        #[derive(khal::ShaderArgs)]
        #[workgroup_size(#wg_x, #wg_y, #wg_z)]
        pub struct #args_struct_name<'a> {
            #(#binding_fields)*
            #(#push_constant_fields)*
        }
    };

    // Generate the call method parameters and body for each binding
    let call_params: Vec<proc_macro2::TokenStream> = bindings
        .iter()
        .map(|b| {
            let name = &b.name;
            let elem_ty = &b.element_type;
            let cfg_attrs = &b.cfg_attrs;

            if b.is_mutable {
                quote! {
                    #(#cfg_attrs)*
                    #name: &'a mut (impl khal::AsGpuSliceMut<#elem_ty>),
                }
            } else {
                quote! {
                    #(#cfg_attrs)*
                    #name: &'a (impl khal::AsGpuSlice<#elem_ty>),
                }
            }
        })
        .collect();

    let call_push_constant_params: Vec<proc_macro2::TokenStream> = push_constants
        .iter()
        .map(|pc| {
            let name = &pc.name;
            let ty = &pc.ty;
            let cfg_attrs = &pc.cfg_attrs;

            quote! {
                #(#cfg_attrs)*
                #name: #ty,
            }
        })
        .collect();

    let args_construction: Vec<proc_macro2::TokenStream> = bindings
        .iter()
        .map(|b| {
            let name = &b.name;
            let cfg_attrs = &b.cfg_attrs;

            if b.is_mutable {
                quote! {
                    #(#cfg_attrs)*
                    #name: #name.as_gpu_slice_mut(),
                }
            } else {
                quote! {
                    #(#cfg_attrs)*
                    #name: #name.as_gpu_slice(),
                }
            }
        })
        .collect();

    let push_constant_construction: Vec<proc_macro2::TokenStream> = push_constants
        .iter()
        .map(|pc| {
            let name = &pc.name;
            let cfg_attrs = &pc.cfg_attrs;

            quote! {
                #(#cfg_attrs)*
                #name,
            }
        })
        .collect();

    // ── CPU dispatch code generation ─────────────────────────────────────
    let func_ident = &func.sig.ident;
    let cpu_dispatch_block =
        cpu::generate_cpu_dispatch_block(&original_params, workgroup_size, func_ident);

    // ── CUDA (nvptx64) kernel entry point generation ─────────────────────
    let func_name_str = func.sig.ident.to_string();

    // Hash the full token stream to produce a deterministic suffix unique to each
    // function body/signature. This avoids name conflicts when two modules define
    // entry point functions with the same name.
    let hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        func.to_token_stream().to_string().hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    };
    let cuda_entry_name = format!("{}_cuda_entry_{}", func_name_str, hash);
    let cuda_entry_ident = syn::Ident::new(&cuda_entry_name, func.sig.ident.span());

    let cuda_entry_block = cuda::generate_cuda_entry_block(
        &original_params,
        &bindings,
        workgroup_size,
        func_ident,
        &cuda_entry_ident,
    );

    let cuda_entry_name_str = &cuda_entry_name;

    // Extract doc attributes from the original function to propagate to the generated struct.
    let doc_attrs: Vec<_> = func
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .collect();

    // ── Wrapper struct and impl ──────────────────────────────────────────
    let wrapper_def = quote! {
        #(#doc_attrs)*
        #[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
        pub struct #struct_name {
            pub function: khal::backend::GpuFunction<#args_struct_name<'static>>,
        }

        #[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
        impl #struct_name {
            /// The kernel entry point name (for SPIR-V/WebGPU).
            pub const ENTRY_POINT: &'static str = #func_name_str;
            /// The kernel entry point name for CUDA (PTX) kernels.
            pub const CUDA_ENTRY_POINT: &'static str = #cuda_entry_name_str;
            /// The full module path where this kernel is defined (set by module_path!() at definition site).
            pub const MODULE_PATH: &'static str = module_path!();
            /// Whether this kernel requires SPIR-V passthrough loading (bypassing naga validation).
            pub const SPIRV_PASSTHROUGH: bool = #spirv_passthrough;

            // Markers for compile-time verification that backend features are enabled
            // on this shader crate. Referenced by `#[derive(Shader)]` to catch missing
            // feature propagation.
            #[doc(hidden)]
            #[cfg(feature = "cpu")]
            pub const __ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU: () = ();
            #[doc(hidden)]
            #[cfg(feature = "cpu-parallel")]
            pub const __ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU_PARALLEL: () = ();
            #[doc(hidden)]
            #[cfg(feature = "cuda")]
            pub const __ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CUDA: () = ();

            /// Creates this kernel by finding its shader file in an embedded directory.
            ///
            /// For SPIR-V/WebGPU backends, loads the `.spv` file.
            /// For CUDA backends, loads the `.ptx` file.
            pub fn from_dir(
                backend: &khal::backend::GpuBackend,
                dir: &khal::re_exports::include_dir::Dir<'static>,
            ) -> Result<Self, khal::backend::GpuBackendError> {
                match backend.target() {
                    khal::backend::CompileTarget::Ptx => {
                        Self::from_dir_ptx(backend, dir, Self::CUDA_ENTRY_POINT)
                    }
                    _ => {
                        Self::from_dir_with_entry_point(backend, dir, Self::ENTRY_POINT)
                    }
                }
            }

            /// Loads a kernel from a PTX file in an embedded directory.
            ///
            /// The PTX file contains all kernels from the crate; individual
            /// functions are extracted by entry point name.
            pub fn from_dir_ptx(
                backend: &khal::backend::GpuBackend,
                dir: &khal::re_exports::include_dir::Dir<'static>,
                entry_point: &str,
            ) -> Result<Self, khal::backend::GpuBackendError> {
                // PTX compilation produces a single file for the entire crate.
                let file = dir.get_file("shaders.ptx")
                    .unwrap_or_else(|| panic!("PTX file 'shaders.ptx' not found in embedded dir"));
                Self::from_bytes(backend, file.contents(), entry_point)
            }

            /// Like from_dir, but with an explicit entry point name (SPIR-V path).
            pub fn from_dir_with_entry_point(
                backend: &khal::backend::GpuBackend,
                dir: &khal::re_exports::include_dir::Dir<'static>,
                entry_point: &str,
            ) -> Result<Self, khal::backend::GpuBackendError> {
                // Strip crate name from MODULE_PATH to get relative module path.
                let module = Self::MODULE_PATH.split_once("::")
                    .map(|(_, rest)| rest)
                    .unwrap_or("");

                let filename = if module.is_empty() {
                    format!("{}.spv", entry_point)
                } else {
                    format!("{}-{}.spv", module.replace("::", "-"), entry_point)
                };
                let file = dir.get_file(&filename)
                    .unwrap_or_else(|| panic!("SPIR-V file not found in embedded dir: {}", filename));

                let full_entry = if module.is_empty() {
                    entry_point.to_string()
                } else {
                    format!("{}::{}", module, entry_point)
                };

                #[cfg(target_arch = "wasm32")]
                let full_entry = full_entry.replace("::", "_");

                Self::from_bytes(backend, file.contents(), &full_entry)
            }

            /// Creates this kernel from raw SPIR-V bytes with an explicit entry point.
            pub fn from_bytes(
                backend: &khal::backend::GpuBackend,
                bytes: &[u8],
                entry_point: &str,
            ) -> Result<Self, khal::backend::GpuBackendError> {
                Ok(Self {
                    function: khal::backend::GpuFunction::from_bytes_with_passthrough(
                        backend, bytes, entry_point, Self::SPIRV_PASSTHROUGH,
                    )?,
                })
            }

            pub fn call<'a>(
                &self,
                // Use __ prefix to avoid potential clashes with the arguments names.
                __pass: &mut khal::backend::GpuPass,
                __dispatch_grid: impl Into<khal::backend::DispatchGrid<'a, khal::backend::GpuBackend>>,
                #(#call_params)*
                #(#call_push_constant_params)*
            ) -> Result<(), khal::backend::GpuBackendError> {
                use khal::AsGpuSlice as _;
                use khal::AsGpuSliceMut as _;
                // Eagerly convert the grid so that the DispatchGrid<'a> value can be
                // subtyped to DispatchGrid<'local> when passed to launch_grid.
                let __dispatch_grid = __dispatch_grid.into();
                #cpu_dispatch_block
                let args = #args_struct_name {
                    #(#args_construction)*
                    #(#push_constant_construction)*
                };
                self.function.launch_grid(__pass, &args, __dispatch_grid)
            }
        }
    };

    // Output the original function, the internal args struct, the wrapper struct,
    // and the CUDA kernel entry point (on nvptx64).
    let output = quote! {
        #func
        #args_struct_def
        #wrapper_def
        #cuda_entry_block
    };

    output.into()
}
