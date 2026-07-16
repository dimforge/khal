# Running rust-gpu `#[spirv]` shaders on native CUDA via cuda-oxide

> Status: working prototype (2026-06-05). `gpu_add` and `gemm_tiled` run **bit-exact**
> on the RTX 5090 (sm_120) through khal's CUDA backend, compiled by cuda-oxide.
> A `#[spirv_kernel]` proc-macro auto-lowers the *verbatim* shader source.
>
> **Update — the upstream `--target nvptx64` fix removes the per-shader shims.**
> The original prototype below compiled the device crate on the *host* target, so
> `cfg(target_arch="nvptx64")` was false — which forced a string of workarounds
> (glam `scalar-math`, `khal_std` barrier → `cuda_device::sync_threads`,
> `Vec4::new` for `Vec4::ZERO`). cuda-oxide now supports building a **device-only
> crate for the real `nvptx64-nvidia-cuda` target** (`cargo-oxide build --device`),
> so the cfg is correct and the *verbatim* `khal_std` `link_llvm_intrinsics`
> barrier lowers to a real `bar.sync`. See "Upstream fix" at the bottom. The
> shim story below is kept for the unified-crate (host-target) path.

## The problem

The whole stack (vortx tensors, nexus3d physics) is written once in **rust-gpu
`#[spirv]`** and compiled to SPIR-V for the **WebGPU/Vulkan** backend. khal also
has a **CUDA backend** (`khal/src/backend/cuda.rs`, cudarc-based) — but its
shader→PTX compiler, `cargo-cuda` → `rustc_codegen_nvvm`, needs **LLVM 7**, which
is effectively unbuildable on modern systems. So `--features cuda` could never
get PTX, and the CUDA backend sat unused.

## The fix: cuda-oxide as the PTX compiler

[cuda-oxide](https://github.com/NVlabs/cuda-oxide) is a modern **Rust → PTX**
backend (LLVM 21). It replaces the dead step: compile the kernels with cuda-oxide,
load the PTX/cubin through khal's existing `cudarc` backend. Same vortx ops, now
on native CUDA.

Three layers had to line up:

### 1. ABI — already compatible

khal dispatches a kernel by **entry-point name** with positional `u64` params,
sorted by `(binding.space, binding.index)`:

| `#[spirv(...)]` param | khal passes | cuda-oxide `#[kernel]` lowering |
|---|---|---|
| `storage_buffer` `&[f32]` / `&mut [f32]` | `(device_ptr, byte_len)` | `&[f32]` / `DisjointSlice<f32>` → `(ptr, len)` |
| `uniform` `&Shape` | `(device_ptr)` | `&Shape` → `(ptr)` |
| `push_constant` | raw bytes | scalar params |

cuda-oxide's `#[kernel] fn(a: &[f32])` already lowers to `(ptr, i64)` — the same
shape khal sends. The entry-point name the host loads is
`Self::CUDA_ENTRY_POINT` = `format!("{fn}_cuda_entry_{hash}")` (from
`spirv_bindgen`); read it from a probe crate depending on `vortx-shaders`.

### 2. libdevice — one patch + a cubin step

Arithmetic kernels (GEMM) load as **PTX text** directly. But `exp`/ELU lower to
`__nv_expf` (libdevice), whose PTX has an unresolved extern that the driver JIT
rejects. Fix:

- **Patch khal's loader** (`backend/cuda.rs::load_module_bytes`): detect a CUBIN
  (ELF magic `\x7fELF`) → `cudarc::nvrtc::Ptx::from_binary(bytes)`, else PTX text
  via `from_src`. `cuModuleLoadData` auto-detects the format.
- **Produce a self-contained cubin** from cuda-oxide's `.ll`:
  `cuda_host::ltoir::build_cubin_from_ll(ll, "sm_120")` (libNVVM + libdevice +
  nvJitLink). Needs CUDA 12.8+ `libnvvm`/`libnvJitLink` (pip wheels
  `nvidia-cuda-nvcc-cu12` / `nvidia-nvjitlink-cu12`, pointed at via
  `LIBNVVM_PATH` / `LIBNVJITLINK_PATH`) and the cuda-oxide nightly.

A combined module (GEMM PTX + ELU libdevice) is built as **one cubin** — load it
as `shaders.ptx`; khal's patched loader takes it as a cubin.

> The same wall hits TMA + FMA: TMA's nvvm intrinsic lowers only on the direct-PTX
> path, FMA (`mul_add`) only on the libdevice path — they're mutually exclusive in
> cuda-oxide today.

### 3. Auto-port — compile the existing `#[spirv]` source unchanged

The big realisation: **don't re-author kernels**. rust-gpu shaders are
`spirv-std`-flavored Rust; cuda-oxide speaks `cuda-device`. But:

- `khal-std`, `glamx`, `spirv-std` **already compile for the nvptx target** under
  cuda-oxide (`khal-std` is `#![cfg_attr(target_arch="nvptx64", …)]` by design).
- The shader **body** compiles unchanged (grid-stride loops, `khal_std`
  `at_mut`/`read`, `Shape` math, `glamx` vectors).
- Only the **entry signature** needs a mechanical transform → the `#[spirv_kernel]`
  proc-macro (`crates/spirv2cuda_macros`, or `~/spirv2cuda_macros` on the box).

#### The `#[spirv_kernel]` macro

Apply it where `#[spirv(compute(...))]` would go; the `#[spirv(...)]` *param*
attributes stay in place. It emits a `#[cuda_device::kernel]` fn:

| input | output |
|---|---|
| `#[spirv(global/local_invocation_id\|workgroup_id)] id: UVec3` | dropped param; `let id = UVec3::new(thread::index_1d().get() as u32, 0, 0)` (etc.) in a prelude |
| `#[spirv(uniform)] x: &T` | kept (`&T` → ptr) |
| `#[spirv(storage_buffer)] x: &[f32]` | kept (`(ptr, len)`) |
| `#[spirv(storage_buffer)] x: &mut [f32]` | `DisjointSlice<f32>` param + `let mut x = MutBuf(x)` |
| body | emitted **unchanged** |

The mutable mapping is the crux: cuda-oxide **cannot write through `&mut [T]`
params** (both `a[i]=v` — "Deref→Index assignment not yet implemented" — and
`get_unchecked_mut` stores silently no-op). Its working mutable type is
`DisjointSlice`. So the macro maps mutable storage buffers to `DisjointSlice` and
provides a tiny glue:

```rust
pub struct MutBuf<'a>(pub DisjointSlice<'a, f32>);
impl<'a> MaybeIndexUnchecked<f32> for MutBuf<'a> {
    fn at_mut(&mut self, i: usize) -> &mut f32 { unsafe { self.0.get_unchecked_mut(i) } }
    fn write(&mut self, i: usize, v: f32)      { unsafe { *self.0.get_unchecked_mut(i) = v; } }
    // gpu_add never reads its output buffer; DisjointSlice has no &self read.
    fn read(&self, _i: usize) -> f32 { 0.0 }
    fn at(&self, _i: usize) -> &f32 { static Z: f32 = 0.0; &Z } // (&T->&mut T cast is now a hard error)
}
```

So the body's verbatim `*a.at_mut(ia) += b.read(ib)` lands on a persisting store.

#### Proof

`examples/autoport2` is the **byte-for-byte vortx `gpu_add`** — signature, body,
and `#[spirv(...)]` attributes all in place — under `#[spirv_kernel]`. It compiles
to PTX, loads through khal's CUDA backend, and returns `a + b` with **0.0 error**
over 120 elements. No wrapper written by hand.

## Results (RTX 5090)

- **Ops** (bit-exact vs WebGPU/CPU): `gpu_add` (0.0), `gemm_tiled` (0.0, boundary-
  tested), `elu` (5.96e-8, libdevice cubin), and vortx's own `OpAssign::launch`
  host op on cuda-oxide PTX (0.0).
- **GEMM ladder** (1024³): naive 7.4 → register-blocked 10.5 → +FMA 16.3 →
  +double-buffer **22.8 TFLOPS**.
- **Policy forward** (actor+critic), WebGPU vs cuda-oxide: **5.5–13×** faster
  (0.34 vs 4.5 ms/step at N=8192; 14.4 vs ~1.1 TFLOPS) — WebGPU is dispatch-bound.
- **Hybrid full iteration** (physics stays WebGPU, policy + PPO update on
  cuda-oxide): **1.44–1.52×** end-to-end — the update (3.2×) is the real lever;
  the forward is ~2% of the physics-bound rollout.

## What's left

The mechanism is proven; the rest is **bounded extension**, not research:

1. **Macro coverage** for cases `gpu_add` doesn't hit:
   `#[spirv(workgroup)] &mut [T; N]` → `SharedArray`; push-constant cfg branches;
   more thread builtins; non-`f32` element types; **atomics** (the physics solver).
2. **A real cuda-oxide fix** (optional): support write-back through `&mut [T]`
   params (the mature `rustc_codegen_nvvm` does) — would remove the `MutBuf`
   mapping. Also unsupported: `core::slice::from_raw_parts_mut`.
3. **Zero-copy hybrid transfer**: Vulkan↔CUDA external-memory interop
   (`VK_KHR_external_memory` ↔ `cudaImportExternalMemory`) via wgpu-hal raw
   handles, to erase the per-step obs copy.
4. **Full port**: apply the macro across all vortx + nexus shaders → a fully
   native-CUDA pipeline (no cross-API), at which point the hybrid's transfer wart
   disappears.

## Upstream fix — compile device crates for the real `nvptx64` target

The prototype above worked around a single root cause: `cargo-oxide` ran plain
`cargo build` (host target), so `cfg(target_arch="nvptx64")` was **false** and
every nvptx-gated code path (glam scalar math, `khal_std`'s `bar.sync` barrier,
indexing) silently took its host fallback. The proper fix is to compile the
**device crate** for `nvptx64-nvidia-cuda`. Four small cuda-oxide changes make
this a first-class mode — and then the *verbatim* shader source compiles with
**no shims at all**:

1. **cargo-oxide** — new flag `cargo-oxide build --device [--arch sm_120] <crate>`
   adds `--target nvptx64-nvidia-cuda -Z build-std=core` (the nvptx target ships
   no precompiled `core`) and points `CUDA_OXIDE_PTX_DIR` at the crate dir.
2. **cuda-macros** — gate the host-side `cuda_host::CudaKernel` marker behind
   `#[cfg(not(target_arch="nvptx64"))]`, so the same `#[kernel]` source compiles
   both as a unified host+device crate *and* as a pure device crate (where
   `cuda_host` isn't in scope).
3. **mir-importer** — honor `#[link_name="llvm.*"]` externs (the
   `link_llvm_intrinsics` pattern rust-gpu / `khal_std` use). rustc resolves such
   a foreign item to a symbol whose mangled name *is* the link name, so the
   importer surfaces `llvm.nvvm.barrier0` as the dispatch name and routes it to
   the same `Barrier0Op` as `cuda_device::sync_threads`. General fix: any crate's
   `llvm.nvvm.*` intrinsic externs now lower instead of failing PTX verification
   with "Symbol … not found".
4. **rustc-codegen-cuda** — skip the host-object *embed* step for an nvptx
   `llvm_target` (a device-only crate has no host object to embed into); the
   standalone `<crate>.ptx` is the deliverable.

**Proof** (`examples/nvptx_probe`): a `#![no_std]` device-only crate whose kernel
uses the *verbatim* `khal_std` barrier and bakes `cfg!(target_arch="nvptx64")`
into its output. `cargo-oxide build --device --arch sm_120 nvptx_probe` →
PTX with `bar.sync` present and the cfg marker = nvptx-true → runs on the 5090
through khal/cuda with **0.0 error**. The per-shader workarounds are gone.

**Bridge to the full vortx stack — closed.** `khal_std`'s nvptx path used to
depend on `cuda_std` (Rust-CUDA), which transitively pulls `half`, and `half`
fails under `-Z build-std` for nvptx. `cuda_std` was used only for thread-index
sregs and `GpuFloat` math. khal_std now has a **`cuda-oxide` feature** (alongside
the default `rust-cuda`) that:

- declares the 12 `%tid`/`%ctaid`/`%ntid`/`%nctaid` reads as khal_std's own
  `llvm.nvvm.read.ptx.sreg.*` externs (lowered by the importer's link-name
  honoring — the same mechanism as the barrier; the importer also gained
  `llvm.nvvm.read.ptx.sreg.*` aliases on its existing sreg dispatch arms);
- routes `Float` to `num_traits` (libm) instead of `cuda_std::float::GpuFloat`;
- drops the `cuda_std` dependency entirely, so the device crate builds under
  `-Z build-std`.

(It also needs `#![feature(asm_experimental_arch)]` on nvptx for the warp-shuffle
`asm!` in `sync::subgroup`.) Build shader crates with
`khal-std = { default-features = false, features = ["cuda-oxide", …] }`.

**Capstone proof** (`examples/gemm_device`): the *verbatim* vortx `gemm_tiled`
as a `#![no_std]` device-only crate — verbatim `khal_std` barrier (no
`sync_threads` shim), glamx `nostd-libm` (no `scalar-math` override),
khal_std `cuda-oxide` feature. `cargo-oxide build --device --arch sm_120
gemm_device` → `bar.sync:2`, two `.shared` arrays, `.entry gemm_tiled` → runs on
the 5090 (100×50 @ 50×70, boundary-tested) with **0.0 error**. Every per-shader
workaround is gone. Remaining: `elu` (libdevice / `__nv_expf` cubin path) and the
rest of the vortx kernels, then nexus physics (atomics).

## Where things live (on the baguette box)

- cuda-oxide toolchain: `~/cuda-oxide-src`, LLVM 21 at `~/llvm21`, nightly
  `2026-04-03`, `cargo-oxide` on PATH. Env per build: `CUDA_OXIDE_LLC`,
  `LIBCLANG_PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`, and for libdevice
  `LIBNVVM_PATH` / `LIBNVJITLINK_PATH` (12.9 wheels in `~/nvvm-wheel`,
  `~/nvjit-wheel`).
- proc-macro: `~/spirv2cuda_macros`. cubin tool: `~/make_cubin`.
- examples: `~/cuda-oxide-src/crates/rustc-codegen-cuda/examples/{autoport2,
  khal_gemm,khal_elu,khal_fwd,khal_opassign,…}`. khal host demos: `~/khal_*_demo`,
  `~/fwd_bench`, `~/upd_bench`, `~/vortx_cuda_host`.
- patches (local, baguette only): `khal/src/backend/cuda.rs` (cubin loader),
  `khal-builder/src/lib.rs` (`build_ptx` injects cuda-oxide PTX, skipping
  cargo-cuda). Backups in `/tmp/*.bak`. WebGPU training is unaffected (these only
  fire under `--features cuda`).
