use quote::quote;

use super::{BuiltinKind, OriginalParam, OriginalParamKind, ShaderBinding, is_slice_reference};

/// Generate the CUDA (nvptx64) kernel entry point.
///
/// This function:
/// 1. Receives raw device pointers + element counts as u64 parameters
///    (the host's `CudaDispatch` pushes `byte_len / size_of::<T>()` for slice
///    bindings, the same `(ptr, len)` ABI cuda-oxide's `&[T]` params use)
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
    // storage buffers become (ptr: u64, len: u64), uniforms become (ptr: u64).
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
        let len_name = syn::Ident::new(&format!("{}_len", name), name.span());

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
            // Builtins derive from the VIRTUAL workgroup of the current loop
            // iteration (see `virtual_workgroup_loop`), not the hardware block.
            OriginalParamKind::Builtin(kind) => {
                let arg = match kind {
                    BuiltinKind::GlobalInvocationId => {
                        quote! { __khal_wg * __khal_block_dim + __khal_tid }
                    }
                    BuiltinKind::LocalInvocationId => quote! { __khal_tid },
                    BuiltinKind::WorkgroupId => quote! { __khal_wg },
                    BuiltinKind::NumWorkgroups => quote! { __khal_num_wg },
                    BuiltinKind::LocalInvocationIndex => {
                        quote! {
                            (__khal_tid.z * #wg_x * #wg_y + __khal_tid.y * #wg_x + __khal_tid.x)
                        }
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
                let len_name = syn::Ident::new(&format!("{}_len", name), name.span());
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
                                    #len_name as usize,
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
                                    #len_name as usize,
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
        let (loop_prologue, wg_decl, iteration_barrier) = virtual_workgroup_loop(quote! {
            if __khal_indirect_len >= 3 {
                let __a = unsafe {
                    core::slice::from_raw_parts(__khal_indirect_ptr as *const u32, 3)
                };
                Some([__a[0], __a[1], __a[2]])
            } else {
                None
            }
        });
        quote! {
            #[cfg(all(target_arch = "nvptx64", not(feature = "cuda-oxide")))]
            #[khal_std::cuda_std::kernel]
            pub unsafe fn #cuda_entry_ident(
                #(#cuda_params,)*
                __khal_indirect_ptr: u64,
                __khal_indirect_len: u64,
            ) {
                #(#cuda_body)*
                #loop_prologue
                while __khal_lin < __khal_total {
                    #wg_decl
                    #func_ident(#(#cuda_call_args),*);
                    __khal_lin += __khal_stride;
                    #iteration_barrier
                }
            }
        }
    } else {
        quote! {}
    }
}

/// Generate a CUDA entry point for the **cuda-oxide** PTX backend.
///
/// cuda-oxide kernels take TYPED params (`&T` uniforms, `&[T]`/`&mut [T]`
/// storage buffers — khal passes each as `(ptr, len)`), so this inlines the
/// shader body directly: bindings become kernel params, builtins are computed
/// from `khal_std::arch::cuda`, and `#[spirv(workgroup)] &mut [f32; N]` becomes
/// a function-local `SharedArray` wrapped in `SmemBuf` (shared memory). The
/// body runs verbatim against those bindings.
pub(super) fn generate_cuda_oxide_entry_block(
    func: &syn::ItemFn,
    original_params: &[OriginalParam],
    bindings: &[ShaderBinding],
    workgroup_size: [u32; 3],
    cuda_entry_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let wg_x = workgroup_size[0];
    let wg_y = workgroup_size[1];
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

    // Push-constant kernels have no uniform fallback here -> skip.
    if all_params
        .iter()
        .any(|p| matches!(p.kind, OriginalParamKind::PushConstant))
    {
        return quote! {};
    }

    // Kernel params = bindings, in (descriptor_set, binding) order, typed.
    let mut sorted_bindings: Vec<&OriginalParam> = all_params
        .iter()
        .filter(|p| matches!(p.kind, OriginalParamKind::Binding { .. }))
        .copied()
        .collect();
    sorted_bindings.sort_by_key(|p| {
        bindings
            .iter()
            .find(|b| b.name == p.name)
            .map(|b| (b.descriptor_set, b.binding))
            .unwrap_or((0, 0))
    });
    // The host (khal CudaDispatch) pushes a `(ptr, len)` pair for EVERY
    // storage binding. A `&[T]`/`&mut [T]` typed param already lowers to that
    // 2-arg ABI, but a sized-array ref `&[T; N]`/`&mut [T; N]` lowers to a single
    // thin pointer — so the kernel would consume one fewer arg than the host
    // pushes, shifting every later binding's pointer by a slot (a following
    // buffer pointer becomes a length value → illegal-address on first access).
    // Receive array storage bindings as SLICES here (matching the 2-arg ABI) and
    // reconstruct the `&[T; N]`/`&mut [T; N]` in a prelude below.
    let mut array_reconstructions: Vec<proc_macro2::TokenStream> = Vec::new();
    let kernel_params: Vec<proc_macro2::TokenStream> = sorted_bindings
        .iter()
        .map(|p| {
            let name = &p.name;
            let ty = &p.ty;
            // Uniform bindings are pushed by the host as a single pointer, so keep
            // them typed (thin pointer ABI). Storage bindings are pushed as a
            // `(ptr, len)` pair: slice refs already match that 2-arg ABI, but
            // sized-array refs (`&[T; N]`) and scalar refs (`&T`) lower to a single
            // thin pointer and would under-consume the host's args, shifting every
            // later binding. Receive those as slices and reconstruct the original
            // reference in a prelude so the ABI stays aligned.
            let is_uniform = matches!(
                p.kind,
                OriginalParamKind::Binding {
                    is_uniform: true,
                    ..
                }
            );
            if is_uniform {
                return quote! { #name: #ty };
            }
            if let syn::Type::Reference(r) = ty {
                match &*r.elem {
                    // slice storage already lowers to (ptr, len)
                    syn::Type::Slice(_) => quote! { #name: #ty },
                    // sized array storage: receive as slice, rebuild `[T; N]` ref
                    syn::Type::Array(arr) => {
                        let elem_ty = &*arr.elem;
                        let len_expr = &arr.len;
                        let buf_name = syn::Ident::new(&format!("{}_buf", name), name.span());
                        if r.mutability.is_some() {
                            array_reconstructions.push(quote! {
                                let #name: &mut [#elem_ty; #len_expr] = unsafe {
                                    &mut *(#buf_name.as_mut_ptr() as *mut [#elem_ty; #len_expr])
                                };
                            });
                            quote! { #buf_name: &mut [#elem_ty] }
                        } else {
                            array_reconstructions.push(quote! {
                                let #name: &[#elem_ty; #len_expr] = unsafe {
                                    &*(#buf_name.as_ptr() as *const [#elem_ty; #len_expr])
                                };
                            });
                            quote! { #buf_name: &[#elem_ty] }
                        }
                    }
                    // scalar storage (`&T` / `&mut T`): receive as slice, rebuild ref
                    scalar => {
                        let elem_ty = scalar;
                        let buf_name = syn::Ident::new(&format!("{}_buf", name), name.span());
                        if r.mutability.is_some() {
                            array_reconstructions.push(quote! {
                                let #name: &mut #elem_ty =
                                    unsafe { &mut *#buf_name.as_mut_ptr() };
                            });
                            quote! { #buf_name: &mut [#elem_ty] }
                        } else {
                            array_reconstructions.push(quote! {
                                let #name: &#elem_ty = unsafe { &*#buf_name.as_ptr() };
                            });
                            quote! { #buf_name: &[#elem_ty] }
                        }
                    }
                }
            } else {
                quote! { #name: #ty }
            }
        })
        .collect();

    // Preludes for workgroup shared memory (once per kernel) and for builtins
    // (once per virtual-workgroup iteration), in original order.
    let mut preludes: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut builtin_preludes: Vec<proc_macro2::TokenStream> = Vec::new();
    for p in &all_params {
        let name = &p.name;
        match &p.kind {
            // Builtins derive from the VIRTUAL workgroup of the current loop
            // iteration (see `virtual_workgroup_loop`), not the hardware block.
            OriginalParamKind::Builtin(kind) => {
                let expr = match kind {
                    BuiltinKind::GlobalInvocationId => {
                        quote! { __khal_wg * __khal_block_dim + __khal_tid }
                    }
                    BuiltinKind::LocalInvocationId => quote! { __khal_tid },
                    BuiltinKind::WorkgroupId => quote! { __khal_wg },
                    BuiltinKind::NumWorkgroups => quote! { __khal_num_wg },
                    // Flattened like SPIR-V's LocalInvocationIndex
                    // (x + y * size_x + z * size_x * size_y): kernels declared
                    // with 2D/3D `threads(..)` are launched with 2D/3D blocks.
                    BuiltinKind::LocalInvocationIndex => {
                        quote! {
                            (__khal_tid.z * #wg_x * #wg_y + __khal_tid.y * #wg_x + __khal_tid.x)
                        }
                    }
                    _ => quote! { Default::default() },
                };
                builtin_preludes.push(quote! { let #name = #expr; });
            }
            OriginalParamKind::Workgroup => {
                // `#[spirv(workgroup)] x: &mut [T; N]`  ->  a function-local
                // `SharedArray` (shared address space) viewed as a real
                // `&mut [T; N]`; and `x: &mut T` (scalar broadcast slot) ->
                // `SharedArray<T, 1>` viewed as `&mut T`. A real reference (not a
                // wrapper) so the body can index it directly AND pass it to
                // helpers that take `&mut [T; N]` / `&mut T`.
                let static_name = syn::Ident::new(&format!("__smem_{}", name), name.span());
                if let syn::Type::Reference(r) = &p.ty {
                    match &*r.elem {
                        // `&mut [f32; N]` shared tile -> a real `&mut [f32; N]` into
                        // the `SharedArray`'s storage, so the body can index it,
                        // call MaybeIndexUnchecked on it AND pass it verbatim to
                        // helpers typed `&mut [T; N]` (e.g. tree reductions).
                        // `SharedArray::as_mut_ptr` is intercepted by the cuda-oxide
                        // importer and yields a generic pointer to shared memory
                        // (cvta.shared), so accesses through the reference are
                        // ordinary generic loads/stores on shared memory.
                        syn::Type::Array(arr) => {
                            let elem_ty = &*arr.elem;
                            let len_expr = &arr.len;
                            preludes.push(quote! {
                                static mut #static_name:
                                    khal_std::cuda_oxide_glue::SharedArray<#elem_ty, { #len_expr }> =
                                    khal_std::cuda_oxide_glue::SharedArray::UNINIT;
                                let #name: &mut [#elem_ty; #len_expr] = unsafe {
                                    &mut *((&mut *core::ptr::addr_of_mut!(#static_name)).as_mut_ptr()
                                        as *mut [#elem_ty; #len_expr])
                                };
                            });
                        }
                        // `&mut T` scalar broadcast slot -> `SharedArray<T,1>` ->
                        // `&mut T` (real shared reference; works through helpers).
                        scalar => {
                            let elem_ty = scalar;
                            let sref_name =
                                syn::Ident::new(&format!("__sref_{}", name), name.span());
                            preludes.push(quote! {
                                static mut #static_name:
                                    khal_std::cuda_oxide_glue::SharedArray<#elem_ty, 1> =
                                    khal_std::cuda_oxide_glue::SharedArray::UNINIT;
                                let #sref_name: &mut khal_std::cuda_oxide_glue::SharedArray<#elem_ty, 1> =
                                    unsafe { &mut *core::ptr::addr_of_mut!(#static_name) };
                                let #name: &mut #elem_ty = &mut #sref_name[0];
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let body = &func.block;

    // Name the entry with cuda-oxide's reserved kernel prefix directly
    // (`KERNEL_PREFIX` in cuda-oxide's `reserved-oxide-symbols`, currently
    // `cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_<entry>`) instead of
    // going through the `#[kernel]` proc-macro: the collector roots kernels by
    // that def-path marker and the PTX entry keeps the unprefixed base name,
    // while the macro's host-side glue (a `cuda_host::CudaKernel` marker impl)
    // would drag a cuda-host dependency into pure shader crates. The prefix
    // must track cuda-oxide's: the backend rejects roots using an older
    // prefix (e.g. the pre-"scoped cache protocol" `cuda_oxide_kernel_246e25db_`)
    // with an explicit diagnostic. Feature-gated, not target-gated: with the
    // cuda-oxide backend installed the crate is compiled on the HOST target
    // (unified interception) or for nvptx64 (device-only builds) — the entry
    // is generated identically for both.
    let prefixed_entry_ident = syn::Ident::new(
        &format!("{CUDA_OXIDE_KERNEL_PREFIX}{cuda_entry_ident}"),
        cuda_entry_ident.span(),
    );
    let (loop_prologue, wg_decl, iteration_barrier) = virtual_workgroup_loop(quote! {
        if __khal_indirect_args.len() >= 3 {
            Some([__khal_indirect_args[0], __khal_indirect_args[1], __khal_indirect_args[2]])
        } else {
            None
        }
    });
    quote! {
        // The "scoped Cargo cache protocol": cuda-oxide's `#[kernel]` records
        // the backend's codegen identity (output mode, arch, tool provenance)
        // in the *device crate's* dep-info so that only crates owning device
        // code are rebuilt when it changes. `option_env!` does the same from
        // stable Rust (rustc tracks env vars read by `env!`/`option_env!`).
        #[cfg(all(feature = "cuda-oxide", not(target_arch = "spirv")))]
        const _: (Option<&str>, Option<&str>, Option<&str>) = (
            option_env!("CUDA_OXIDE_INTERNAL_CODEGEN_FINGERPRINT"),
            option_env!("CUDA_OXIDE_MATERIALIZE_CUBIN"),
            option_env!("CUDA_OXIDE_INTERNAL_MATERIALIZER_PROVENANCE"),
        );

        #[cfg(all(feature = "cuda-oxide", not(target_arch = "spirv")))]
        #[allow(non_snake_case)]
        // `no_mangle` forces local codegen: without it rustc's
        // cross-crate-inlinable heuristic skips emitting a mono item for
        // trivial kernels (nothing in-crate calls an entry point), and the
        // collector never sees them.
        #[unsafe(no_mangle)]
        pub fn #prefixed_entry_ident(
            #(#kernel_params,)*
            __khal_indirect_args: &[u32],
        ) {
            #(#preludes)*
            #(#array_reconstructions)*
            #loop_prologue
            // The shader body runs once per virtual workgroup. It lives in a
            // closure so that an early `return` in the body ends the current
            // iteration, not the whole loop.
            let mut __khal_body = |__khal_wg: khal_std::glamx::UVec3| {
                #(#builtin_preludes)*
                #body
            };
            while __khal_lin < __khal_total {
                #wg_decl
                __khal_body(__khal_wg);
                __khal_lin += __khal_stride;
                #iteration_barrier
            }
        }
    }
}

/// cuda-oxide's reserved kernel-entry prefix (`reserved_oxide_symbols::KERNEL_PREFIX`).
/// The backend strips it to obtain the PTX `.entry` name, so the PTX entry is
/// exactly `CUDA_ENTRY_POINT`. Keep in sync with the `cuda-device` revision
/// pinned in `khal-std/Cargo.toml`.
const CUDA_OXIDE_KERNEL_PREFIX: &str = "cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_";

/// Virtual-workgroup loop shared by both CUDA entry flavours.
///
/// CUDA has no device-side indirect dispatch. khal's CUDA backend therefore
/// launches indirect dispatches with a fixed number of resident blocks and
/// hands the entry the indirect-args `[u32; 3]` as a trailing `(ptr, len)`
/// slice (`(0, 0)` for direct dispatches). The entry loops over the virtual
/// workgroups `lin = block_id; lin < x*y*z; lin += real_grid_size`, so:
/// - a direct dispatch runs exactly one iteration per block (no overhead
///   beyond a few integer ops),
/// - an indirect dispatch runs the exact virtual workgroup count without any
///   host round trip,
/// - the trip count is uniform per block, so workgroup barriers in the body
///   remain valid.
///
/// `read_args` must evaluate to `Option<[u32; 3]>` (the indirect args when
/// present). Returns `(prologue, wg_decl, iteration_barrier)`: the prologue
/// defines `__khal_tid`, `__khal_block_dim`, `__khal_num_wg` and the loop
/// bounds; the per-iteration declaration defines `__khal_wg` from
/// `__khal_lin`; the barrier separates consecutive virtual workgroups.
fn virtual_workgroup_loop(
    read_args: proc_macro2::TokenStream,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let prologue = quote! {
        let __khal_tid = khal_std::arch::cuda::thread_idx();
        let __khal_block_dim = khal_std::arch::cuda::block_dim();
        let __khal_real_grid = khal_std::arch::cuda::num_workgroups();
        let __khal_num_wg: khal_std::glamx::UVec3 = match #read_args {
            Some(a) => khal_std::glamx::UVec3::new(a[0], a[1], a[2]),
            None => __khal_real_grid,
        };
        let __khal_total = __khal_num_wg.x * __khal_num_wg.y * __khal_num_wg.z;
        let __khal_stride = __khal_real_grid.x * __khal_real_grid.y * __khal_real_grid.z;
        let __khal_bid = khal_std::arch::cuda::block_idx();
        let mut __khal_lin = __khal_bid.x
            + __khal_bid.y * __khal_real_grid.x
            + __khal_bid.z * __khal_real_grid.x * __khal_real_grid.y;
    };
    let wg_decl = quote! {
        let __khal_wg = khal_std::glamx::UVec3::new(
            __khal_lin % __khal_num_wg.x,
            (__khal_lin / __khal_num_wg.x) % __khal_num_wg.y,
            __khal_lin / (__khal_num_wg.x * __khal_num_wg.y),
        );
    };
    // Prevents fast threads from overwriting shared memory for the next virtual workgroup while
    // slow threads still read the previous one. The condition is uniform per block, so the
    // barrier is well-formed (and skipped on the last iteration).
    let iteration_barrier = quote! {
        if __khal_lin < __khal_total {
            khal_std::sync::workgroup_memory_barrier_with_group_sync();
        }
    };
    (prologue, wg_decl, iteration_barrier)
}
