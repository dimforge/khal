#[cfg(feature = "webgpu")]
use crate::backend::WebGpu;
#[cfg(feature = "cuda")]
use crate::backend::cuda::{
    Cuda, CudaBackendError, CudaBuffer, CudaBufferSlice, CudaDispatch as CudaDispatchInner,
    CudaEncoder as CudaEncoderInner, CudaFunction as CudaFunctionInner,
    CudaModule as CudaModuleInner, CudaPass as CudaPassInner, CudaTimestamps,
};
#[cfg(feature = "webgpu")]
use crate::backend::webgpu::CommandEncoderExt;
use crate::backend::webgpu::WebGpuTimestamps;
use crate::backend::{
    Backend, BufferUsages, DeviceValue, Dispatch, DispatchGrid, Encoder, MaybeSendSync,
    ShaderBinding,
};
use crate::shader::{ShaderArgsError, ShaderArgsType};
use bytemuck::{AnyBitPattern, NoUninit};
use std::marker::PhantomData;
use std::ops::RangeBounds;
#[cfg(feature = "cpu")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "webgpu")]
use wgpu::ComputePass;

/// Backend-agnostic GPU context that dispatches to the active backend at runtime.
#[non_exhaustive]
#[derive(Clone)]
pub enum GpuBackend {
    #[cfg(feature = "webgpu")]
    WebGpu(WebGpu),
    #[cfg(feature = "cuda")]
    Cuda(Cuda),
    #[cfg(feature = "cpu")]
    Cpu,
}

impl GpuBackend {
    /// Returns the compile target for this backend instance.
    pub fn target(&self) -> super::CompileTarget {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(_) => super::CompileTarget::Wgsl,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => super::CompileTarget::Ptx,
            #[cfg(feature = "cpu")]
            Self::Cpu => super::CompileTarget::Wgsl,
        }
    }

    /// Returns `true` if this is the CUDA backend.
    #[cfg(feature = "cuda")]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(..))
    }

    /// Loads a SPIR-V module using passthrough loading (bypassing naga validation).
    ///
    /// On the WebGPU backend, this uses `create_shader_module_spirv` to pass raw SPIR-V
    /// directly to the Vulkan driver without naga transpilation. This is useful for shaders
    /// that use SPIR-V features not supported by naga (e.g. scalar block layout).
    /// Falls back to normal loading if passthrough is not available.
    ///
    /// On other backends (Vulkan/ash, CUDA), this is equivalent to `load_module_bytes`.
    pub fn load_module_bytes_spirv_passthrough(
        &self,
        bytes: &[u8],
    ) -> Result<GpuModule, GpuBackendError> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => Ok(GpuModule::WebGpu(
                backend.load_module_spirv_passthrough(bytes)?,
            )),
            // For non-WebGPU backends, SPIR-V is already loaded natively.
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => <Self as Backend>::load_module_bytes(self, bytes),
            #[cfg(feature = "cpu")]
            Self::Cpu => <Self as Backend>::load_module_bytes(self, bytes),
        }
    }
}

/// A GPU buffer that wraps the active backend's buffer type.
#[non_exhaustive]
pub enum GpuBuffer<T: DeviceValue> {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Buffer<T>),
    #[cfg(feature = "cuda")]
    Cuda(CudaBuffer<T>),
    #[cfg(feature = "cpu")]
    Cpu(Vec<T>),
}

impl<T: DeviceValue> GpuBuffer<T> {
    /// Returns the underlying CPU slice. Panics if this is not a CPU buffer.
    #[cfg(feature = "cpu")]
    pub fn unwrap_slice(&self) -> &[T] {
        match self {
            Self::Cpu(slice) => slice,
            _ => panic!("cannot unwrap a buffer on backends other than CPU"),
        }
    }

    /// Returns the underlying mutable CPU slice. Panics if this is not a CPU buffer.
    #[cfg(feature = "cpu")]
    pub fn unwrap_slice_mut(&mut self) -> &mut [T] {
        match self {
            Self::Cpu(slice) => slice,
            _ => panic!("cannot unwrap a buffer on backends other than CPU"),
        }
    }

    /// Returns a slice view of the entire buffer.
    pub fn as_slice(&self) -> GpuBufferSlice<'_, T> {
        use crate::backend::Buffer;
        Buffer::<GpuBackend, T>::slice(self, ..)
    }

    /// Returns a mutable slice view of the entire buffer.
    pub fn as_slice_mut(&mut self) -> GpuBufferSliceMut<'_, T> {
        self.slice_mut(..)
    }
}

/// Trait for types that can provide an immutable GPU buffer slice.
///
/// This is used by generated kernel wrapper structs to accept various buffer-like
/// types (e.g., `GpuBuffer<T>`, `Tensor<T>`) as kernel arguments.
pub trait AsGpuSlice<T: DeviceValue> {
    /// Returns an immutable GPU buffer slice.
    fn as_gpu_slice(&self) -> GpuBufferSlice<'_, T>;
}

/// Trait for types that can provide a mutable GPU buffer slice.
///
/// This is used by generated kernel wrapper structs to accept various buffer-like
/// types (e.g., `GpuBuffer<T>`, `Tensor<T>`) as mutable kernel arguments.
pub trait AsGpuSliceMut<T: DeviceValue> {
    /// Returns a mutable GPU buffer slice.
    fn as_gpu_slice_mut(&mut self) -> GpuBufferSliceMut<'_, T>;
}

impl<T: DeviceValue> AsGpuSlice<T> for GpuBuffer<T> {
    fn as_gpu_slice(&self) -> GpuBufferSlice<'_, T> {
        self.as_slice()
    }
}

impl<T: DeviceValue> AsGpuSliceMut<T> for GpuBuffer<T> {
    fn as_gpu_slice_mut(&mut self) -> GpuBufferSliceMut<'_, T> {
        self.as_slice_mut()
    }
}

impl<'a, T: DeviceValue> AsGpuSlice<T> for GpuBufferSlice<'a, T> {
    fn as_gpu_slice(&self) -> GpuBufferSlice<'_, T> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(s) => GpuBufferSlice::WebGpu(*s),
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => GpuBufferSlice::Cuda(*s),
            #[cfg(feature = "cpu")]
            Self::Cpu(s) => GpuBufferSlice::Cpu(s),
        }
    }
}

impl<'a, T: DeviceValue> AsGpuSliceMut<T> for GpuBufferSliceMut<'a, T> {
    fn as_gpu_slice_mut(&mut self) -> GpuBufferSliceMut<'_, T> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(s) => GpuBufferSliceMut::WebGpu(*s),
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => GpuBufferSliceMut::Cuda(*s),
            #[cfg(feature = "cpu")]
            Self::Cpu(s) => GpuBufferSliceMut::Cpu(s),
        }
    }
}

impl<'a> From<&'a GpuBuffer<[u32; 3]>> for DispatchGrid<'a, GpuBackend> {
    fn from(buffer: &'a GpuBuffer<[u32; 3]>) -> Self {
        DispatchGrid::Indirect(buffer)
    }
}

impl<'a> DispatchGrid<'a, GpuBackend> {
    /// Resolves the dispatch grid to concrete workgroup counts.
    ///
    /// - `Grid`: returns the counts directly.
    /// - `ThreadCount`: divides by `workgroup_size` (rounding up).
    /// - `Indirect`: reads the counts from the CPU-backed buffer.
    #[cfg(feature = "cpu")]
    pub fn resolve_to_workgroup_counts(&self, workgroup_size: &[u32; 3]) -> [u32; 3] {
        match self {
            DispatchGrid::Grid(g) => *g,
            DispatchGrid::ThreadCount(t) => [
                t[0].div_ceil(workgroup_size[0]),
                t[1].div_ceil(workgroup_size[1]),
                t[2].div_ceil(workgroup_size[2]),
            ],
            DispatchGrid::Indirect(buf) => buf.unwrap_slice()[0],
        }
    }
}

/// An immutable view into a [`GpuBuffer`].
#[non_exhaustive]
pub enum GpuBufferSlice<'a, T: DeviceValue> {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::BufferSlice<'a, T>),
    #[cfg(feature = "cuda")]
    Cuda(CudaBufferSlice),
    #[cfg(feature = "cpu")]
    Cpu(&'a [T]),
}

impl<'a, T: DeviceValue> GpuBufferSlice<'a, T> {
    /// Returns the underlying CPU slice. Panics if this is not a CPU buffer slice.
    #[cfg(feature = "cpu")]
    pub fn unwrap_slice(&self) -> &[T] {
        match self {
            Self::Cpu(slice) => slice,
            _ => panic!("cannot unwrap a buffer on backends other than CPU"),
        }
    }
}

impl<'a, T: DeviceValue + bytemuck::Pod> GpuBufferSlice<'a, T> {
    /// Reinterprets this buffer slice as a different element type.
    ///
    /// This is safe because the underlying GPU buffer slice doesn't carry type information
    /// at runtime - the type parameter is only for compile-time safety. Both types must
    /// implement `Pod` to ensure safe reinterpretation.
    ///
    /// # Panics
    /// Panics if `size_of::<T>() != size_of::<U>()`.
    pub fn cast<U: DeviceValue + bytemuck::Pod>(self) -> GpuBufferSlice<'a, U> {
        assert_eq!(
            core::mem::size_of::<T>(),
            core::mem::size_of::<U>(),
            "Cannot cast GpuBufferSlice: size_of::<{}>() != size_of::<{}>()",
            core::any::type_name::<T>(),
            core::any::type_name::<U>()
        );
        // For GPU backends (WebGpu, Cuda), the underlying buffer slice doesn't
        // carry type information at runtime - it's just byte offsets. For the CPU backend,
        // we cast the slice using bytemuck which is safe for Pod types of equal size.
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(slice) => GpuBufferSlice::WebGpu(slice),
            #[cfg(feature = "cuda")]
            Self::Cuda(slice) => GpuBufferSlice::Cuda(slice),
            #[cfg(feature = "cpu")]
            Self::Cpu(slice) => GpuBufferSlice::Cpu(bytemuck::cast_slice(slice)),
        }
    }

    /// Reinterprets this buffer slice as a different element type, allowing different
    /// element sizes. The total byte size must be divisible by the target element size.
    ///
    /// Use this when reinterpreting e.g. a `&[f32]` buffer as `&[Vec4]` (4 f32s per Vec4).
    ///
    /// # Panics
    /// Panics if the total byte count is not a multiple of `size_of::<U>()`,
    /// or if either type has zero size.
    pub fn reinterpret<U: DeviceValue + bytemuck::Pod>(self) -> GpuBufferSlice<'a, U> {
        if core::mem::size_of::<T>() == core::mem::size_of::<U>() {
            return self.cast();
        }
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(slice) => {
                let target_size = core::mem::size_of::<U>() as u64;
                assert!(
                    target_size > 0 && slice.byte_len % target_size == 0,
                    "Cannot reinterpret WebGpu buffer: byte length {} is not a multiple of size_of::<{}>()",
                    slice.byte_len,
                    core::any::type_name::<U>()
                );
                GpuBufferSlice::WebGpu(slice)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(slice) => {
                let target_size = core::mem::size_of::<U>() as u64;
                assert!(
                    target_size > 0 && slice.byte_len % target_size == 0,
                    "Cannot reinterpret Cuda buffer: byte length {} is not a multiple of size_of::<{}>()",
                    slice.byte_len,
                    core::any::type_name::<U>()
                );
                GpuBufferSlice::Cuda(slice)
            }
            #[cfg(feature = "cpu")]
            Self::Cpu(slice) => GpuBufferSlice::Cpu(bytemuck::cast_slice(slice)),
        }
    }
}

/// A mutable view into a [`GpuBuffer`].
#[non_exhaustive]
pub enum GpuBufferSliceMut<'a, T: DeviceValue> {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::BufferSlice<'a, T>), // TODO: add a mut version of ::BufferSlice?
    #[cfg(feature = "cuda")]
    Cuda(CudaBufferSlice),
    #[cfg(feature = "cpu")]
    Cpu(&'a mut [T]),
}

impl<'a, T: DeviceValue + bytemuck::Pod> GpuBufferSliceMut<'a, T> {
    /// Reinterprets this mutable buffer slice as a different element type.
    ///
    /// # Panics
    /// Panics if `size_of::<T>() != size_of::<U>()`.
    pub fn cast<U: DeviceValue + bytemuck::Pod>(self) -> GpuBufferSliceMut<'a, U> {
        assert_eq!(
            core::mem::size_of::<T>(),
            core::mem::size_of::<U>(),
            "Cannot cast GpuBufferSliceMut: size_of::<{}>() != size_of::<{}>()",
            core::any::type_name::<T>(),
            core::any::type_name::<U>()
        );
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(slice) => GpuBufferSliceMut::WebGpu(slice),
            #[cfg(feature = "cuda")]
            Self::Cuda(slice) => GpuBufferSliceMut::Cuda(slice),
            #[cfg(feature = "cpu")]
            Self::Cpu(slice) => GpuBufferSliceMut::Cpu(bytemuck::cast_slice_mut(slice)),
        }
    }

    /// Reinterprets this mutable buffer slice as a different element type, allowing different
    /// element sizes. The total byte size must be divisible by the target element size.
    ///
    /// # Panics
    /// Panics if the total byte count is not a multiple of `size_of::<U>()`,
    /// or if either type has zero size.
    pub fn reinterpret<U: DeviceValue + bytemuck::Pod>(self) -> GpuBufferSliceMut<'a, U> {
        if core::mem::size_of::<T>() == core::mem::size_of::<U>() {
            return self.cast();
        }
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(slice) => {
                let target_size = core::mem::size_of::<U>() as u64;
                assert!(
                    target_size > 0 && slice.byte_len % target_size == 0,
                    "Cannot reinterpret WebGpu buffer: byte length {} is not a multiple of size_of::<{}>()",
                    slice.byte_len,
                    core::any::type_name::<U>()
                );
                GpuBufferSliceMut::WebGpu(slice)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(slice) => {
                let target_size = core::mem::size_of::<U>() as u64;
                assert!(
                    target_size > 0 && slice.byte_len % target_size == 0,
                    "Cannot reinterpret Cuda buffer: byte length {} is not a multiple of size_of::<{}>()",
                    slice.byte_len,
                    core::any::type_name::<U>()
                );
                GpuBufferSliceMut::Cuda(slice)
            }
            #[cfg(feature = "cpu")]
            Self::Cpu(slice) => GpuBufferSliceMut::Cpu(bytemuck::cast_slice_mut(slice)),
        }
    }
}

impl<'a, T: DeviceValue> GpuBufferSliceMut<'a, T> {
    /// Returns the underlying mutable CPU slice. Panics if this is not a CPU buffer slice.
    #[cfg(feature = "cpu")]
    pub fn unwrap_slice(&mut self) -> &mut [T] {
        match self {
            Self::Cpu(slice) => slice,
            _ => panic!("cannot unwrap a buffer on backends other than CPU"),
        }
    }
}

/// Backend-agnostic command encoder wrapping the active backend's encoder.
#[non_exhaustive]
pub enum GpuEncoder {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Encoder),
    #[cfg(feature = "cuda")]
    Cuda(CudaEncoderInner),
    #[cfg(feature = "cpu")]
    Cpu,
    Noop,
}

/// Backend-agnostic compute pass wrapping the active backend's pass.
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum GpuPass {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Pass),
    #[cfg(feature = "cuda")]
    Cuda(CudaPassInner),
    #[cfg(feature = "cpu")]
    Cpu(Option<CpuPassTimer>),
    Noop,
}

/// Records the elapsed time of a compute pass on the CPU backend.
///
/// When dropped, computes the duration since creation and appends a
/// [`GpuTimestamp`] entry to the shared results vector.
#[cfg(feature = "cpu")]
pub struct CpuPassTimer {
    label: String,
    start: std::time::Instant,
    entries: Arc<Mutex<Vec<GpuTimestamp>>>,
}

#[cfg(feature = "cpu")]
impl Drop for CpuPassTimer {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.entries.lock().unwrap().push(GpuTimestamp {
            label: std::mem::take(&mut self.label),
            duration_ms,
        });
    }
}

impl GpuPass {
    /// Returns true if this is a CPU pass.
    #[cfg(feature = "cpu")]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(..))
    }

    /// Returns `true` if this is the CUDA backend.
    #[cfg(feature = "cuda")]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(..))
    }

    /// Begins a compute dispatch within this pass, binding the given function.
    pub fn begin_dispatch<'a>(&'a mut self, function: &'a InnerGpuFunction) -> GpuDispatch<'a> {
        match (self, function) {
            #[cfg(feature = "webgpu")]
            (Self::WebGpu(pass), InnerGpuFunction::WebGpu(f)) => {
                GpuDispatch::WebGpu(pass.begin_dispatch(f))
            }
            #[cfg(feature = "cuda")]
            (Self::Cuda(pass), InnerGpuFunction::Cuda(f)) => GpuDispatch::Cuda(CudaDispatchInner {
                stream: &pass.stream,
                function: f,
                args: Vec::new(),
                #[cfg(feature = "push_constants")]
                push_constants: Vec::new(),
            }),
            #[cfg(feature = "cpu")]
            (Self::Cpu(_), InnerGpuFunction::Noop) => GpuDispatch::Noop,
            (Self::Noop, InnerGpuFunction::Noop) => GpuDispatch::Noop,
            _ => panic!("Mismatched pass/function backend types"),
        }
    }
}

/// A loaded shader module wrapping the active backend's module type.
#[non_exhaustive]
pub enum GpuModule {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Module),
    #[cfg(feature = "cuda")]
    Cuda(CudaModuleInner),
    Noop,
}

/// A GPU compute function that can be dispatched.
///
/// The `Args` type parameter specifies the [`ShaderArgsType`] that this function accepts.
/// This enables compile-time verification that the correct arguments are passed when
/// launching the function.
///
/// # Type Parameter
///
/// - `Args`: A marker type implementing [`ShaderArgsType`] that represents the family of
///   ShaderArgs types this function accepts. Use `()` for untyped functions.
///
/// # Example
///
/// ```ignore
/// #[derive(Shader)]
/// pub struct MyShader {
///     // Typed function - only accepts MyKernelArgs
///     my_kernel: GpuFunction<MyKernelArgsType>,
/// }
///
/// // This compiles:
/// shader.my_kernel.launch(backend, pass, &MyKernelArgs { ... }, ...);
///
/// // This would fail to compile:
/// shader.my_kernel.launch(backend, pass, &WrongArgs { ... }, ...);
/// ```
#[non_exhaustive]
pub enum GpuFunction<Args: ShaderArgsType = ()> {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Function, PhantomData<Args>),
    #[cfg(feature = "cuda")]
    Cuda(CudaFunctionInner, PhantomData<Args>),
    Noop(PhantomData<Args>),
}

/// Internal untyped GPU function used by the backend.
///
/// This is the type-erased version of [`GpuFunction`] used internally for
/// backend operations. Users should use [`GpuFunction<Args>`] instead.
#[non_exhaustive]
#[derive(Clone)]
pub enum InnerGpuFunction {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Function),
    #[cfg(feature = "cuda")]
    Cuda(CudaFunctionInner),
    Noop,
}

/// Backend-agnostic dispatch that collects bindings and launches a compute kernel.
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum GpuDispatch<'a> {
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as Backend>::Dispatch<'a>),
    #[cfg(feature = "cuda")]
    Cuda(CudaDispatchInner<'a>),
    Noop,
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<&'a ()>),
}

/// Errors from GPU backend operations, wrapping backend-specific error types.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum GpuBackendError {
    #[cfg(feature = "webgpu")]
    #[error(transparent)]
    WebGpu(#[from] <WebGpu as Backend>::Error),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] CudaBackendError),
    #[error(transparent)]
    ShaderArgs(#[from] ShaderArgsError),
    #[error("GPU context not found in local storage")]
    ContextNotFound,
    #[error("Noop backend error")]
    Noop,
}

/// Result of a single timed compute pass.
#[derive(Clone)]
pub struct GpuTimestamp {
    /// Name of the compute pass that was timed.
    pub label: String,
    /// Elapsed wall-clock duration in milliseconds.
    pub duration_ms: f64,
}

/// CPU timestamp manager.
///
/// Collects per-pass durations measured via [`CpuPassTimer`] on drop.
#[cfg(feature = "cpu")]
pub struct CpuTimestamps {
    entries: Arc<Mutex<Vec<GpuTimestamp>>>,
}

#[cfg(feature = "cpu")]
impl CpuTimestamps {
    fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Backend-agnostic GPU timestamp manager for profiling compute passes.
#[non_exhaustive]
pub enum GpuTimestamps {
    #[cfg(feature = "webgpu")]
    WebGpu(WebGpuTimestamps),
    #[cfg(feature = "cuda")]
    Cuda(CudaTimestamps),
    #[cfg(feature = "cpu")]
    Cpu(CpuTimestamps),
    Noop,
}

impl GpuTimestamps {
    /// Creates a new GPU timestamp manager with the given capacity (number of timed passes).
    ///
    /// Returns `Noop` if the backend doesn't support timestamps.
    pub fn new(backend: &GpuBackend, capacity: u32) -> Self {
        match backend {
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu(webgpu) => WebGpuTimestamps::new(webgpu, capacity)
                .map(GpuTimestamps::WebGpu)
                .unwrap_or(GpuTimestamps::Noop),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => GpuTimestamps::Cuda(CudaTimestamps::new(cuda)),
            #[cfg(feature = "cpu")]
            GpuBackend::Cpu => GpuTimestamps::Cpu(CpuTimestamps::new()),
            #[allow(unreachable_patterns)]
            _ => GpuTimestamps::Noop,
        }
    }

    /// Whether this timestamp manager is active (not Noop).
    pub fn is_enabled(&self) -> bool {
        !matches!(self, GpuTimestamps::Noop)
    }

    /// Resets the timestamp manager for a new frame.
    pub fn reset(&mut self) {
        match self {
            #[cfg(feature = "webgpu")]
            GpuTimestamps::WebGpu(ts) => ts.reset(),
            #[cfg(feature = "cuda")]
            GpuTimestamps::Cuda(ts) => ts.reset(),
            #[cfg(feature = "cpu")]
            GpuTimestamps::Cpu(ts) => ts.entries.lock().unwrap().clear(),
            GpuTimestamps::Noop => {}
        }
    }

    /// Resolves timestamp queries to a buffer and copies to staging for readback.
    ///
    /// Must be called before submitting the encoder. No-op for CPU timestamps
    /// (results are collected on pass drop).
    pub fn resolve(&self, encoder: &mut GpuEncoder) {
        match (self, encoder) {
            #[cfg(feature = "webgpu")]
            (GpuTimestamps::WebGpu(ts), GpuEncoder::WebGpu(enc)) => ts.resolve(enc),
            _ => {}
        }
    }

    /// Reads back timestamp results after GPU synchronization.
    ///
    /// Returns per-pass durations in milliseconds. Call after `backend.synchronize()`.
    pub async fn read(&self, backend: &GpuBackend) -> Result<Vec<GpuTimestamp>, GpuBackendError> {
        match (self, backend) {
            #[cfg(feature = "webgpu")]
            (GpuTimestamps::WebGpu(ts), GpuBackend::WebGpu(webgpu)) => ts.read(webgpu).await,
            #[cfg(feature = "cuda")]
            (GpuTimestamps::Cuda(ts), _) => Ok(ts.read()?),
            #[cfg(feature = "cpu")]
            (GpuTimestamps::Cpu(ts), _) => Ok(ts.entries.lock().unwrap().clone()),
            _ => Ok(Vec::new()),
        }
    }
}

impl Backend for GpuBackend {
    const NAME: &'static str = "any";
    const TARGET: super::CompileTarget = super::CompileTarget::Wgsl;

    type Error = GpuBackendError;
    type Buffer<T: DeviceValue> = GpuBuffer<T>;
    type BufferSlice<'b, T: DeviceValue> = GpuBufferSlice<'b, T>;
    type Encoder = GpuEncoder;
    type Pass = GpuPass;
    type Timestamps = GpuTimestamps;
    type Module = GpuModule;
    type Function = InnerGpuFunction;
    type Dispatch<'a> = GpuDispatch<'a>;

    /*
     * Module/function loading.
     */
    fn load_module(&self, data: &str) -> Result<Self::Module, Self::Error> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => Ok(GpuModule::WebGpu(backend.load_module(data)?)),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => Ok(GpuModule::Cuda(backend.load_module(data)?)),
            #[cfg(feature = "cpu")]
            Self::Cpu => Ok(GpuModule::Noop),
        }
    }

    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => Ok(GpuModule::WebGpu(backend.load_module_bytes(bytes)?)),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => Ok(GpuModule::Cuda(backend.load_module_bytes(bytes)?)),
            #[cfg(feature = "cpu")]
            Self::Cpu => Ok(GpuModule::Noop),
        }
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
    ) -> Result<Self::Function, Self::Error> {
        match (self, module) {
            #[cfg(feature = "webgpu")]
            (Self::WebGpu(backend), GpuModule::WebGpu(module)) => Ok(InnerGpuFunction::WebGpu(
                backend.load_function(module, entry_point, push_constant_size)?,
            )),
            #[cfg(feature = "cuda")]
            (Self::Cuda(backend), GpuModule::Cuda(module)) => Ok(InnerGpuFunction::Cuda(
                backend.load_function(module, entry_point, push_constant_size)?,
            )),
            #[cfg(feature = "cpu")]
            (Self::Cpu, GpuModule::Noop) => Ok(InnerGpuFunction::Noop),
            _ => panic!("Invalid backend/module type pair"),
        }
    }

    fn load_function_with_layouts(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
        layouts: &crate::shader::BindGroupLayoutInfo,
    ) -> Result<Self::Function, Self::Error> {
        match (self, module) {
            #[cfg(feature = "webgpu")]
            (Self::WebGpu(backend), GpuModule::WebGpu(module)) => Ok(InnerGpuFunction::WebGpu(
                backend.load_function_with_layouts(
                    module,
                    entry_point,
                    push_constant_size,
                    layouts,
                )?,
            )),
            #[cfg(feature = "cuda")]
            (Self::Cuda(backend), GpuModule::Cuda(module)) => {
                Ok(InnerGpuFunction::Cuda(backend.load_function_with_layouts(
                    module,
                    entry_point,
                    push_constant_size,
                    layouts,
                )?))
            }
            #[cfg(feature = "cpu")]
            (Self::Cpu, GpuModule::Noop) => Ok(InnerGpuFunction::Noop),
            _ => panic!("Invalid backend/module type pair"),
        }
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        match self {
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => GpuEncoder::WebGpu(backend.begin_encoding()),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => GpuEncoder::Cuda(backend.begin_encoding()),
            #[cfg(feature = "cpu")]
            Self::Cpu => GpuEncoder::Cpu,
        }
    }

    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> GpuDispatch<'a> {
        match (self, pass, function) {
            #[cfg(feature = "webgpu")]
            (Self::WebGpu(backend), GpuPass::WebGpu(pass), InnerGpuFunction::WebGpu(function)) => {
                GpuDispatch::WebGpu(backend.begin_dispatch(pass, function))
            }
            #[cfg(feature = "cuda")]
            (Self::Cuda(backend), GpuPass::Cuda(pass), InnerGpuFunction::Cuda(function)) => {
                GpuDispatch::Cuda(backend.begin_dispatch(pass, function))
            }
            #[cfg(feature = "cpu")]
            (Self::Cpu, GpuPass::Cpu(_), InnerGpuFunction::Noop) => GpuDispatch::Noop,
            (_, GpuPass::Noop, InnerGpuFunction::Noop) => GpuDispatch::Noop,
            _ => panic!("Invalid backend/pass/function type triple"),
        }
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        match (self, encoder) {
            #[cfg(feature = "webgpu")]
            (Self::WebGpu(backend), GpuEncoder::WebGpu(encoder)) => Ok(backend.submit(encoder)?),
            #[cfg(feature = "cuda")]
            (Self::Cuda(backend), GpuEncoder::Cuda(encoder)) => Ok(backend.submit(encoder)?),
            #[cfg(feature = "cpu")]
            (Self::Cpu, GpuEncoder::Cpu) => Ok(()),
            _ => panic!("Invalid backend/encoder type pair"),
        }
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + NoUninit>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu(backend) => Ok(GpuBuffer::WebGpu(backend.init_buffer(data, usage)?)),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(backend) => Ok(GpuBuffer::Cuda(backend.init_buffer(data, usage)?)),
            #[cfg(feature = "cpu")]
            GpuBackend::Cpu => Ok(GpuBuffer::Cpu(data.to_vec())),
        }
    }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu(backend) => {
                Ok(GpuBuffer::WebGpu(backend.uninit_buffer::<T>(len, usage)?))
            }
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(backend) => {
                Ok(GpuBuffer::Cuda(backend.uninit_buffer::<T>(len, usage)?))
            }
            #[cfg(feature = "cpu")]
            GpuBackend::Cpu => {
                let mut v = Vec::with_capacity(len);
                // SAFETY: T: DeviceValue + NoUninit, so zeroed memory is valid.
                v.resize(len, unsafe { std::mem::zeroed() });
                Ok(GpuBuffer::Cpu(v))
            }
        }
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        match (self, buffer) {
            #[cfg(feature = "webgpu")]
            (GpuBackend::WebGpu(backend), GpuBuffer::WebGpu(buffer)) => {
                backend.write_buffer(buffer, offset, data)?
            }
            #[cfg(feature = "cuda")]
            (GpuBackend::Cuda(backend), GpuBuffer::Cuda(buffer)) => {
                backend.write_buffer(buffer, offset, data)?
            }
            #[cfg(feature = "cpu")]
            (GpuBackend::Cpu, GpuBuffer::Cpu(buffer)) => {
                let start = offset as usize;
                buffer[start..start + data.len()].copy_from_slice(data);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Invalid backend/buffer type pair"),
        }

        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu(backend) => Ok(backend.synchronize()?),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(backend) => Ok(backend.synchronize()?),
            #[cfg(feature = "cpu")]
            GpuBackend::Cpu => Ok(()),
        }
    }

    async fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        match (self, buffer) {
            #[cfg(feature = "webgpu")]
            (GpuBackend::WebGpu(backend), GpuBuffer::WebGpu(buffer)) => {
                backend.read_buffer(buffer, out).await?
            }
            #[cfg(feature = "cuda")]
            (GpuBackend::Cuda(backend), GpuBuffer::Cuda(buffer)) => {
                backend.read_buffer(buffer, out).await?
            }
            #[cfg(feature = "cpu")]
            (GpuBackend::Cpu, GpuBuffer::Cpu(buffer)) => {
                out[..buffer.len()].copy_from_slice(buffer);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Invalid backend/buffer type pair"),
        }
        Ok(())
    }

    async fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        match (self, buffer) {
            #[cfg(feature = "webgpu")]
            (GpuBackend::WebGpu(backend), GpuBuffer::WebGpu(buffer)) => {
                backend.slow_read_buffer(buffer, out).await?
            }
            #[cfg(feature = "cuda")]
            (GpuBackend::Cuda(backend), GpuBuffer::Cuda(buffer)) => {
                backend.slow_read_buffer(buffer, out).await?
            }
            #[cfg(feature = "cpu")]
            (GpuBackend::Cpu, GpuBuffer::Cpu(buffer)) => {
                out[..buffer.len()].copy_from_slice(buffer);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Invalid backend/buffer type pair"),
        }
        Ok(())
    }
}

impl Encoder<GpuBackend> for GpuEncoder {
    fn begin_pass(&mut self, label: &str, timestamps: Option<&mut GpuTimestamps>) -> GpuPass {
        match self {
            #[cfg(feature = "webgpu")]
            GpuEncoder::WebGpu(encoder) => {
                if let Some(GpuTimestamps::WebGpu(ts)) = timestamps {
                    GpuPass::WebGpu(encoder.begin_pass(label, Some(ts)))
                } else {
                    GpuPass::WebGpu(encoder.begin_pass(label, None))
                }
            }
            #[cfg(feature = "cuda")]
            GpuEncoder::Cuda(encoder) => {
                if let Some(GpuTimestamps::Cuda(ts)) = timestamps {
                    GpuPass::Cuda(encoder.begin_pass(label, Some(ts)))
                } else {
                    GpuPass::Cuda(encoder.begin_pass(label, None))
                }
            }
            #[cfg(feature = "cpu")]
            GpuEncoder::Cpu => {
                let timer = if let Some(GpuTimestamps::Cpu(ts)) = timestamps {
                    Some(CpuPassTimer {
                        label: label.to_string(),
                        start: std::time::Instant::now(),
                        entries: ts.entries.clone(),
                    })
                } else {
                    None
                };
                GpuPass::Cpu(timer)
            }
            GpuEncoder::Noop => GpuPass::Noop,
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &<GpuBackend as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <GpuBackend as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), GpuBackendError> {
        match (self, source, target) {
            #[cfg(feature = "webgpu")]
            (GpuEncoder::WebGpu(encoder), GpuBuffer::WebGpu(source), GpuBuffer::WebGpu(target)) => {
                Encoder::<WebGpu>::copy_buffer_to_buffer::<T>(
                    encoder,
                    source,
                    source_offset,
                    target,
                    target_offset,
                    copy_len,
                )?;
            }
            #[cfg(feature = "cuda")]
            (GpuEncoder::Cuda(encoder), GpuBuffer::Cuda(source), GpuBuffer::Cuda(target)) => {
                Encoder::<Cuda>::copy_buffer_to_buffer::<T>(
                    encoder,
                    source,
                    source_offset,
                    target,
                    target_offset,
                    copy_len,
                )?;
            }
            #[cfg(feature = "cpu")]
            (GpuEncoder::Cpu, GpuBuffer::Cpu(source), GpuBuffer::Cpu(target)) => {
                target[target_offset..target_offset + copy_len]
                    .copy_from_slice(&source[source_offset..source_offset + copy_len]);
            }
            _ => panic!("Invalid encoder/buffer type combination"),
        }
        Ok(())
    }
}

impl<'a> Dispatch<'a, GpuBackend> for GpuDispatch<'a> {
    #[cfg(feature = "push_constants")]
    fn set_push_constants(&mut self, data: &[u8]) {
        match self {
            #[cfg(feature = "webgpu")]
            GpuDispatch::WebGpu(dispatch) => dispatch.set_push_constants(data),
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(dispatch) => dispatch.set_push_constants(data),
            GpuDispatch::Noop => {}
            GpuDispatch::_Phantom(_) => unreachable!(),
        }
    }

    // NOTE: the block_dim is configured in the shader…
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, GpuBackend>>,
        block_dim: [u32; 3],
    ) -> Result<(), GpuBackendError> {
        match self {
            #[cfg(feature = "webgpu")]
            GpuDispatch::WebGpu(dispatch) => {
                let grid: DispatchGrid<'b, GpuBackend> = grid.into();
                let webgpu_grid = match grid {
                    DispatchGrid::Grid(dims) => DispatchGrid::Grid(dims),
                    DispatchGrid::ThreadCount(threads) => DispatchGrid::ThreadCount(threads),
                    DispatchGrid::Indirect(buffer) => match buffer {
                        GpuBuffer::WebGpu(buf) => DispatchGrid::Indirect(buf),
                        #[allow(unreachable_patterns)]
                        _ => panic!("Invalid buffer type for WebGpu dispatch"),
                    },
                };
                dispatch.launch(webgpu_grid, block_dim)?;
            }
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(dispatch) => {
                let grid: DispatchGrid<'b, GpuBackend> = grid.into();
                let cuda_grid = match grid {
                    DispatchGrid::Grid(dims) => DispatchGrid::Grid(dims),
                    DispatchGrid::ThreadCount(threads) => DispatchGrid::ThreadCount(threads),
                    DispatchGrid::Indirect(buffer) => match buffer {
                        GpuBuffer::Cuda(buf) => DispatchGrid::Indirect(buf),
                        _ => panic!("Invalid buffer type for Cuda dispatch"),
                    },
                };
                dispatch.launch(cuda_grid, block_dim)?;
            }
            GpuDispatch::Noop => {}
            GpuDispatch::_Phantom(_) => unreachable!(),
        }
        Ok(())
    }
}

#[cfg(feature = "webgpu")]
impl CommandEncoderExt for GpuEncoder {
    fn compute_pass<'encoder>(
        &'encoder mut self,
        label: &str,
        // timestamps: Option<&mut GpuTimestamps>,
    ) -> ComputePass<'encoder> {
        match self {
            GpuEncoder::WebGpu(encoder) => encoder.compute_pass(label),
            #[cfg(feature = "cuda")]
            GpuEncoder::Cuda(_) => panic!("Cannot create compute pass from non-WebGpu encoder"),
            #[cfg(feature = "cpu")]
            GpuEncoder::Cpu => panic!("Cannot create compute pass from non-WebGpu encoder"),
            GpuEncoder::Noop => panic!("Cannot create compute pass from non-WebGpu encoder"),
        }
    }
}

impl<'b, T: DeviceValue> crate::ShaderArgs<'b> for GpuBuffer<T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match (self, dispatch) {
            #[cfg(feature = "webgpu")]
            (GpuBuffer::WebGpu(buffer), GpuDispatch::WebGpu(dispatch)) => {
                dispatch.args.push((
                    binding,
                    super::webgpu::WebGpuBufferSlice {
                        byte_len: buffer.size(),
                        inner: buffer.slice(..),
                    },
                ));
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (GpuBuffer::Cuda(buffer), GpuDispatch::Cuda(dispatch)) => {
                dispatch.set_arg(binding, buffer.device_ptr_raw(), buffer.byte_len());
                Ok(())
            }
            #[cfg(feature = "cpu")]
            (GpuBuffer::Cpu(_), GpuDispatch::Noop) => Ok(()),
            _ => panic!("Invalid buffer/dispatch type combination"),
        }
    }
}

impl<'b, T: DeviceValue> crate::ShaderArgs<'b> for GpuBufferSlice<'_, T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match (self, dispatch) {
            #[cfg(feature = "webgpu")]
            (GpuBufferSlice::WebGpu(slice), GpuDispatch::WebGpu(dispatch)) => {
                dispatch.args.push((binding, *slice));
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (GpuBufferSlice::Cuda(slice), GpuDispatch::Cuda(dispatch)) => {
                dispatch.set_arg(binding, slice.offset_ptr(), slice.byte_len);
                Ok(())
            }
            #[cfg(feature = "cpu")]
            (GpuBufferSlice::Cpu(_), GpuDispatch::Noop) => Ok(()),
            _ => panic!("Invalid buffer slice/dispatch type combination"),
        }
    }
}

impl<'b, T: DeviceValue> crate::ShaderArgs<'b> for GpuBufferSliceMut<'_, T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match (self, dispatch) {
            #[cfg(feature = "webgpu")]
            (GpuBufferSliceMut::WebGpu(slice), GpuDispatch::WebGpu(dispatch)) => {
                dispatch.args.push((binding, *slice));
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (GpuBufferSliceMut::Cuda(slice), GpuDispatch::Cuda(dispatch)) => {
                dispatch.set_arg(binding, slice.offset_ptr(), slice.byte_len);
                Ok(())
            }
            #[cfg(feature = "cpu")]
            (GpuBufferSliceMut::Cpu(_), GpuDispatch::Noop) => Ok(()),
            _ => panic!("Invalid mutable buffer slice/dispatch type combination"),
        }
    }
}

impl<T: DeviceValue> GpuBuffer<T> {
    /// Returns a mutable slice view of the given element range.
    // TODO: have this part of the trait?
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> GpuBufferSliceMut<'_, T> {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBuffer::WebGpu(buffer) => {
                use crate::backend::Buffer;
                GpuBufferSliceMut::WebGpu(Buffer::<WebGpu, T>::slice(buffer, range))
            }
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda(buffer) => {
                use crate::backend::Buffer;
                GpuBufferSliceMut::Cuda(Buffer::<Cuda, T>::slice(buffer, range))
            }
            #[cfg(feature = "cpu")]
            GpuBuffer::Cpu(buffer) => {
                use std::ops::Bound;
                let start = match range.start_bound() {
                    Bound::Included(&n) => n,
                    Bound::Excluded(&n) => n + 1,
                    Bound::Unbounded => 0,
                };
                let end = match range.end_bound() {
                    Bound::Included(&n) => n + 1,
                    Bound::Excluded(&n) => n,
                    Bound::Unbounded => buffer.len(),
                };
                GpuBufferSliceMut::Cpu(&mut buffer[start..end])
            }
        }
    }
}

impl<T: DeviceValue> crate::backend::Buffer<GpuBackend, T> for GpuBuffer<T> {
    fn is_empty(&self) -> bool {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBuffer::WebGpu(buffer) => buffer.size() == 0,
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda(buffer) => {
                use crate::backend::Buffer;
                Buffer::<Cuda, T>::is_empty(buffer)
            }
            #[cfg(feature = "cpu")]
            GpuBuffer::Cpu(buffer) => buffer.is_empty(),
        }
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBuffer::WebGpu(buffer) => buffer.size() as usize / std::mem::size_of::<T>(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda(buffer) => {
                use crate::backend::Buffer;
                Buffer::<Cuda, T>::len(buffer)
            }
            #[cfg(feature = "cpu")]
            GpuBuffer::Cpu(buffer) => buffer.len(),
        }
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> GpuBufferSlice<'_, T> {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBuffer::WebGpu(buffer) => {
                use crate::backend::Buffer;
                GpuBufferSlice::WebGpu(Buffer::<WebGpu, T>::slice(buffer, range))
            }
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda(buffer) => {
                use crate::backend::Buffer;
                GpuBufferSlice::Cuda(Buffer::<Cuda, T>::slice(buffer, range))
            }
            #[cfg(feature = "cpu")]
            GpuBuffer::Cpu(buffer) => {
                use std::ops::Bound;
                let start = match range.start_bound() {
                    Bound::Included(&n) => n,
                    Bound::Excluded(&n) => n + 1,
                    Bound::Unbounded => 0,
                };
                let end = match range.end_bound() {
                    Bound::Included(&n) => n + 1,
                    Bound::Excluded(&n) => n,
                    Bound::Unbounded => buffer.len(),
                };
                GpuBufferSlice::Cpu(&buffer[start..end])
            }
        }
    }

    /// Returns a slice of the entire buffer.
    fn as_slice(&self) -> GpuBufferSlice<'_, T> {
        self.slice(..)
    }

    fn usage(&self) -> BufferUsages {
        match self {
            #[cfg(feature = "webgpu")]
            GpuBuffer::WebGpu(buffer) => buffer.usage().into(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda(buffer) => {
                use crate::backend::Buffer;
                Buffer::<Cuda, T>::usage(buffer)
            }
            #[cfg(feature = "cpu")]
            GpuBuffer::Cpu(_) => BufferUsages::all(), // CPU buffers have no usage restrictions
        }
    }
}
