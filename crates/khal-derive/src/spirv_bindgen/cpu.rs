use quote::quote;

use super::{BuiltinKind, OriginalParam, OriginalParamKind, is_slice_reference};

/// Data collected for a single CPU dispatch variant.
struct VariantBody {
    extractions: Vec<proc_macro2::TokenStream>,
    workgroup_inits: Vec<proc_macro2::TokenStream>,
    has_workgroup: bool,
    // For threaded mode (shared memory kernels):
    mut_ptr_setup: Vec<proc_macro2::TokenStream>,
    wg_ptr_setup: Vec<proc_macro2::TokenStream>,
    thread_restore: Vec<proc_macro2::TokenStream>,
    threaded_call_args: Vec<proc_macro2::TokenStream>,
}

/// Generate extraction + call args for a given set of active params.
fn generate_variant_body(
    active_params: &[&OriginalParam],
    workgroup_size: [u32; 3],
) -> VariantBody {
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];
    let wg_z = workgroup_size[2];

    let mut body = VariantBody {
        extractions: vec![],
        workgroup_inits: vec![],
        has_workgroup: false,
        mut_ptr_setup: vec![],
        wg_ptr_setup: vec![],
        thread_restore: vec![],
        threaded_call_args: vec![],
    };

    for param in active_params {
        let name = &param.name;
        match &param.kind {
            OriginalParamKind::Builtin(kind) => {
                let arg = match kind {
                    BuiltinKind::GlobalInvocationId => {
                        quote! {
                            khal_std::glamx::UVec3::new(
                                __wg_x * #wg_x + __tx,
                                __wg_y * #wg_y + __ty,
                                __wg_z * #wg_z + __tz,
                            )
                        }
                    }
                    BuiltinKind::LocalInvocationId => {
                        quote! {
                            khal_std::glamx::UVec3::new(__tx, __ty, __tz)
                        }
                    }
                    BuiltinKind::WorkgroupId => {
                        quote! {
                            khal_std::glamx::UVec3::new(__wg_x, __wg_y, __wg_z)
                        }
                    }
                    BuiltinKind::NumWorkgroups => {
                        quote! {
                            khal_std::glamx::UVec3::new(__grid[0], __grid[1], __grid[2])
                        }
                    }
                    BuiltinKind::LocalInvocationIndex => {
                        quote! {
                            (__tz * #wg_x * #wg_y + __ty * #wg_x + __tx)
                        }
                    }
                    _ => {
                        quote! { Default::default() }
                    }
                };
                body.threaded_call_args.push(arg);
            }
            OriginalParamKind::Binding {
                is_uniform: _,
                is_mutable,
            } => {
                let is_slice = is_slice_reference(&param.ty);
                if *is_mutable {
                    let buf_name = syn::Ident::new(&format!("__cpu_{}_buf", name), name.span());
                    body.extractions.push(quote! {
                        let mut #buf_name = #name.as_gpu_slice_mut();
                        let #name = #buf_name.unwrap_slice();
                    });

                    body.threaded_call_args.push(if is_slice {
                        quote! { #name }
                    } else {
                        quote! { &mut #name[0] }
                    });

                    // For threaded mode: extract raw pointer (as usize for Send) + len
                    let ptr_name = syn::Ident::new(&format!("__{}_ptr", name), name.span());
                    let len_name = syn::Ident::new(&format!("__{}_len", name), name.span());
                    body.mut_ptr_setup.push(quote! {
                        let #ptr_name = #name.as_mut_ptr() as usize;
                        let #len_name = #name.len();
                    });
                    // Inside each thread: reconstruct &mut [T] from raw pointer
                    body.thread_restore.push(quote! {
                            let #name = unsafe { core::slice::from_raw_parts_mut(#ptr_name as *mut _, #len_name) };
                        });
                } else {
                    let slice_name = syn::Ident::new(&format!("__cpu_{}_slice", name), name.span());
                    body.extractions.push(quote! {
                        let #slice_name = #name.as_gpu_slice();
                        let #name = #slice_name.unwrap_slice();
                    });
                    body.threaded_call_args.push(if is_slice {
                        quote! { #name }
                    } else {
                        quote! { &#name[0] }
                    });
                }
            }
            OriginalParamKind::PushConstant => {
                body.threaded_call_args.push(quote! { &#name });
            }
            OriginalParamKind::Workgroup => {
                body.has_workgroup = true;
                let inner_ty = if let syn::Type::Reference(ref_type) = &param.ty {
                    &*ref_type.elem
                } else {
                    &param.ty
                };
                let shared_name = syn::Ident::new(&format!("__cpu_{}_shared", name), name.span());
                let ptr_name = syn::Ident::new(&format!("__{}_wg_ptr", name), name.span());
                body.workgroup_inits.push(quote! {
                    let mut #shared_name: #inner_ty = unsafe { core::mem::zeroed() };
                });
                // For threaded mode: use raw pointer (as usize for Send) to shared memory
                body.wg_ptr_setup.push(quote! {
                    let #ptr_name = core::ptr::addr_of_mut!(#shared_name) as usize;
                });
                body.threaded_call_args
                    .push(quote! { unsafe { &mut *(#ptr_name as *mut _) } });
            }
        }
    }
    body
}

/// Generate the inner dispatch loop for a variant body.
///
/// Non-shared-memory kernels: rayon across workgroups, sequential within each.
/// Shared-memory kernels: sequential workgroup loop, thread pool within each
///   (barriers require actual OS threads for correctness of prefix-sum algorithms).
fn gen_dispatch_loop(
    body: &VariantBody,
    workgroup_size: [u32; 3],
    func_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];
    let wg_z = workgroup_size[2];

    let extractions = &body.extractions;
    let workgroup_inits = &body.workgroup_inits;
    let mut_ptr_setup = &body.mut_ptr_setup;
    let thread_restore = &body.thread_restore;
    let threaded_call_args = &body.threaded_call_args;
    let func = func_ident;

    if body.has_workgroup {
        // Shared-memory kernels: rayon across workgroups, coroutines within each.
        // Each rayon thread runs its own set of coroutines (COROUTINE_YIELDER is
        // thread-local, so they don't interfere).
        let wg_ptr_setup = &body.wg_ptr_setup;

        quote! {
            #(#extractions)*
            #(#mut_ptr_setup)*
            let __total_threads = (#wg_x as usize) * (#wg_y as usize) * (#wg_z as usize);
            khal_std::arch::cpu::dispatch_workgroups(
                (__grid[0] as usize) * (__grid[1] as usize) * (__grid[2] as usize),
                |__wg_flat| {
                    let __wg_x = __wg_flat % __grid[0];
                    let __wg_y = (__wg_flat / __grid[0]) % __grid[1];
                    let __wg_z = __wg_flat / (__grid[0] * __grid[1]);
                    #(#workgroup_inits)*
                    #(#wg_ptr_setup)*
                    khal_std::arch::cpu::dispatch_workgroup_threads(__total_threads, |__flat| {
                        let __flat = __flat;
                        let __tx = __flat % #wg_x;
                        let __ty = (__flat / #wg_x) % #wg_y;
                        let __tz = __flat / (#wg_x * #wg_y);
                        #(#thread_restore)*
                        #func(#(#threaded_call_args),*);
                    });
                },
            );
        }
    } else {
        // Non-shared-memory kernels: rayon across workgroups, sequential within each.
        quote! {
            #(#extractions)*
            #(#mut_ptr_setup)*
            khal_std::arch::cpu::dispatch_workgroups(
                (__grid[0] as usize) * (__grid[1] as usize) * (__grid[2] as usize),
                |__wg_flat| {
                    let __wg_x = __wg_flat % __grid[0];
                    let __wg_y = (__wg_flat / __grid[0]) % __grid[1];
                    let __wg_z = __wg_flat / (__grid[0] * __grid[1]);
                    #(#thread_restore)*
                    for __tz in 0..#wg_z { for __ty in 0..#wg_y { for __tx in 0..#wg_x {
                        #func(#(#threaded_call_args),*);
                    }}}
                },
            );
        }
    }
}

/// Generate the complete CPU dispatch block for a kernel.
///
/// Produces a `#[cfg(feature = "cpu")]` block that, when the pass is CPU-backed,
/// extracts slices from GPU buffers, computes built-in values, and dispatches
/// workgroups via rayon.
pub(super) fn generate_cpu_dispatch_block(
    original_params: &[OriginalParam],
    workgroup_size: [u32; 3],
    func_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];
    let wg_z = workgroup_size[2];

    // Collect unique non-empty cfg attr sets from params (for cfg-variant dispatch).
    let mut cfg_variant_sets: Vec<Vec<syn::Attribute>> = vec![];
    for param in original_params {
        if !param.cfg_attrs.is_empty() {
            let already_exists = cfg_variant_sets.iter().any(|existing| {
                existing.len() == param.cfg_attrs.len()
                    && existing
                        .iter()
                        .zip(param.cfg_attrs.iter())
                        .all(|(a, b)| quote!(#a).to_string() == quote!(#b).to_string())
            });
            if !already_exists {
                cfg_variant_sets.push(param.cfg_attrs.clone());
            }
        }
    }

    // Generate variant blocks. For each unique cfg attr set, we generate a complete
    // extraction + loop + call block wrapped in that cfg, including only params that
    // are always present (no cfg) or matching the current cfg variant.
    if cfg_variant_sets.is_empty() {
        // No cfg-gated params: single variant with all params.
        let all_params: Vec<&OriginalParam> = original_params.iter().collect();
        let body = generate_variant_body(&all_params, workgroup_size);
        let dispatch_loop = gen_dispatch_loop(&body, workgroup_size, func_ident);

        quote! {
            #[cfg(feature = "cpu")]
            {
                if __pass.is_cpu() {
                    let __wg_size = [#wg_x, #wg_y, #wg_z];
                    let __grid = __dispatch_grid.resolve_to_workgroup_counts(&__wg_size);
                    #dispatch_loop
                    return Ok(());
                }
            }
        }
    } else {
        // There are cfg-gated params. Generate one block per unique cfg variant.
        // Each block includes always-present params + params matching that cfg variant.
        let variant_blocks: Vec<proc_macro2::TokenStream> = cfg_variant_sets
            .iter()
            .map(|cfg_attrs| {
                // A param is "active" if it has no cfg attrs or its cfg attrs match this variant.
                let active_params: Vec<&OriginalParam> = original_params
                    .iter()
                    .filter(|p| {
                        p.cfg_attrs.is_empty() || {
                            p.cfg_attrs.len() == cfg_attrs.len()
                                && p.cfg_attrs
                                    .iter()
                                    .zip(cfg_attrs.iter())
                                    .all(|(a, b)| quote!(#a).to_string() == quote!(#b).to_string())
                        }
                    })
                    .collect();

                let body = generate_variant_body(&active_params, workgroup_size);
                let dispatch_loop = gen_dispatch_loop(&body, workgroup_size, func_ident);

                quote! {
                    #(#cfg_attrs)*
                    {
                        #dispatch_loop
                    }
                }
            })
            .collect();

        quote! {
            #[cfg(feature = "cpu")]
            {
                if __pass.is_cpu() {
                    let __wg_size = [#wg_x, #wg_y, #wg_z];
                    let __grid = __dispatch_grid.resolve_to_workgroup_counts(&__wg_size);
                    #(#variant_blocks)*
                    return Ok(());
                }
            }
        }
    }
}
