use quote::quote;

use super::{BuiltinKind, OriginalParam, OriginalParamKind, ShaderBinding, is_slice_reference};

/// Generate the CUDA (nvptx64) kernel entry point.
///
/// This function:
/// 1. Receives raw device pointers + byte lengths as u64 parameters
/// 2. Computes builtin values from CUDA thread/block indices
/// 3. Reconstructs slices from raw pointers
/// 4. Calls the original shader function
pub(super) fn generate_cuda_entry_block(
    original_params: &[OriginalParam],
    bindings: &[ShaderBinding],
    workgroup_size: [u32; 3],
    func_ident: &syn::Ident,
    cuda_entry_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];

    // Generate kernel parameters: for each binding (sorted by set/index),
    // storage buffers become (ptr: u64, byte_len: u64), uniforms become (ptr: u64).
    let mut cuda_params = Vec::new();
    let mut cuda_body = Vec::new();
    let mut cuda_call_args = Vec::new();

    // For the CUDA entry point, include params that are always present or that
    // match the non-push_constants configuration (the default for CUDA builds).
    // Params gated by #[cfg(feature = "push_constants")] are excluded.
    let all_params: Vec<&OriginalParam> = original_params
        .iter()
        .filter(|p| {
            p.cfg_attrs.is_empty()
                || p.cfg_attrs.iter().any(|a| {
                    let s = quote::quote!(#a).to_string();
                    s.contains("not") && s.contains("push_constants")
                })
        })
        .collect();

    // First, collect binding params sorted by (descriptor_set, binding)
    // to match the host-side CudaDispatch parameter order.
    let mut sorted_bindings: Vec<&OriginalParam> = all_params
        .iter()
        .filter(|p| matches!(p.kind, OriginalParamKind::Binding { .. }))
        .copied()
        .collect();
    // Find binding info for sorting
    sorted_bindings.sort_by_key(|p| {
        bindings
            .iter()
            .find(|b| b.name == p.name)
            .map(|b| (b.descriptor_set, b.binding))
            .unwrap_or((0, 0))
    });

    // Generate cuda kernel parameters for bindings
    for param in &sorted_bindings {
        let name = &param.name;
        let ptr_name = syn::Ident::new(&format!("{}_ptr", name), name.span());
        let len_name = syn::Ident::new(&format!("{}_byte_len", name), name.span());

        if let OriginalParamKind::Binding {
            is_uniform,
            is_mutable: _,
        } = &param.kind
        {
            cuda_params.push(quote! { #ptr_name: u64 });
            if !is_uniform {
                cuda_params.push(quote! { #len_name: u64 });
            }
        }
    }

    // Generate cuda kernel parameters for push constants
    let sorted_push_constants: Vec<&OriginalParam> = all_params
        .iter()
        .filter(|p| matches!(p.kind, OriginalParamKind::PushConstant))
        .copied()
        .collect();

    for param in &sorted_push_constants {
        let name = &param.name;
        let inner_ty = if let syn::Type::Reference(ref_type) = &param.ty {
            &*ref_type.elem
        } else {
            &param.ty
        };
        cuda_params.push(quote! { #name: #inner_ty });
    }

    // Generate body: reconstruct slices and compute builtins
    for param in &all_params {
        let name = &param.name;
        match &param.kind {
            OriginalParamKind::Builtin(kind) => {
                let arg = match kind {
                    BuiltinKind::GlobalInvocationId => {
                        quote! { khal_std::arch::cuda::global_invocation_id() }
                    }
                    BuiltinKind::LocalInvocationId => {
                        quote! { khal_std::arch::cuda::local_invocation_id() }
                    }
                    BuiltinKind::WorkgroupId => {
                        quote! { khal_std::arch::cuda::workgroup_id() }
                    }
                    BuiltinKind::NumWorkgroups => {
                        quote! { khal_std::arch::cuda::num_workgroups() }
                    }
                    BuiltinKind::LocalInvocationIndex => {
                        quote! {{
                            let __tid = khal_std::arch::cuda::thread_idx();
                            __tid.z * #wg_x * #wg_y + __tid.y * #wg_x + __tid.x
                        }}
                    }
                    _ => {
                        quote! { Default::default() }
                    }
                };
                cuda_call_args.push(arg);
            }
            OriginalParamKind::Binding {
                is_uniform,
                is_mutable,
            } => {
                let ptr_name = syn::Ident::new(&format!("{}_ptr", name), name.span());
                let len_name = syn::Ident::new(&format!("{}_byte_len", name), name.span());
                let is_slice = is_slice_reference(&param.ty);

                let elem_ty = if let syn::Type::Reference(ref_type) = &param.ty {
                    if let syn::Type::Slice(slice_type) = &*ref_type.elem {
                        &*slice_type.elem
                    } else {
                        &*ref_type.elem
                    }
                } else {
                    &param.ty
                };

                let (body_stmt, call_arg) = if *is_uniform {
                    (
                        quote! { let #name = unsafe { &*(#ptr_name as *const #elem_ty) }; },
                        quote! { #name },
                    )
                } else if *is_mutable && is_slice {
                    (
                        quote! {
                            let #name = unsafe {
                                core::slice::from_raw_parts_mut(
                                    #ptr_name as *mut #elem_ty,
                                    #len_name as usize / core::mem::size_of::<#elem_ty>(),
                                )
                            };
                        },
                        quote! { #name },
                    )
                } else if *is_mutable {
                    (
                        quote! { let #name = unsafe { &mut *(#ptr_name as *mut #elem_ty) }; },
                        quote! { #name },
                    )
                } else if is_slice {
                    (
                        quote! {
                            let #name = unsafe {
                                core::slice::from_raw_parts(
                                    #ptr_name as *const #elem_ty,
                                    #len_name as usize / core::mem::size_of::<#elem_ty>(),
                                )
                            };
                        },
                        quote! { #name },
                    )
                } else {
                    (
                        quote! { let #name = unsafe { &*(#ptr_name as *const #elem_ty) }; },
                        quote! { #name },
                    )
                };

                cuda_body.push(body_stmt);
                cuda_call_args.push(call_arg);
            }
            OriginalParamKind::PushConstant => {
                cuda_call_args.push(quote! { &#name });
            }
            OriginalParamKind::Workgroup => {
                let inner_ty = if let syn::Type::Reference(ref_type) = &param.ty {
                    &*ref_type.elem
                } else {
                    &param.ty
                };
                // Shared memory: use the same UnsafeCell<MaybeUninit<T>> pattern as
                // cuda_std::shared_array! to place in per-block shared memory (.shared
                // address space). A plain `static mut` with address_space(shared) triggers
                // an ICE in rustc_codegen_nvvm.
                let shared_name = syn::Ident::new(&format!("__cuda_{}_shared", name), name.span());
                let wrapper_name = syn::Ident::new(&format!("__CudaShared_{}", name), name.span());
                cuda_body.push(quote! {
                    struct #wrapper_name(core::cell::UnsafeCell<core::mem::MaybeUninit<#inner_ty>>);
                    unsafe impl Send for #wrapper_name {}
                    unsafe impl Sync for #wrapper_name {}
                    #[khal_std::cuda_std::address_space(shared)]
                    static #shared_name: #wrapper_name = #wrapper_name(
                        core::cell::UnsafeCell::new(core::mem::MaybeUninit::uninit())
                    );
                    let #name = unsafe { &mut *(#shared_name.0.get() as *mut #inner_ty) };
                });
                cuda_call_args.push(quote! { #name });
            }
        }
    }

    // Only generate if there are bindings (otherwise it's not a real kernel)
    if !sorted_bindings.is_empty() || !sorted_push_constants.is_empty() {
        quote! {
            #[cfg(target_arch = "nvptx64")]
            #[khal_std::cuda_std::kernel]
            pub unsafe fn #cuda_entry_ident(
                #(#cuda_params),*
            ) {
                #(#cuda_body)*
                #func_ident(#(#cuda_call_args),*);
            }
        }
    } else {
        quote! {}
    }
}
