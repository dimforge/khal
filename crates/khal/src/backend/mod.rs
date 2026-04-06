use crate::shader::{BindGroupLayoutInfo, ShaderArgsError};
use bytemuck::{AnyBitPattern, NoUninit};
use std::error::Error;
use std::ops::RangeBounds;

/// Shader compilation target for different backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileTarget {
    /// WebGPU WGSL shader language
    Wgsl,
    /// NVIDIA PTX (Parallel Thread Execution) assembly
    Ptx,
    /// SPIR-V binary format for Vulkan
    Spirv,
}

#[cfg(feature = "webgpu")]
pub use webgpu::WebGpu;
#[cfg(feature = "webgpu")]
mod webgpu;

#[cfg(feature = "cuda")]
pub use cuda::Cuda;
#[cfg(feature = "cuda")]
pub mod cuda;

mod any_backend;
pub use any_backend::*;

bitflags::bitflags! {
    /// Buffer usage flags that mirror wgpu::BufferUsages.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsages: u32 {
        const MAP_READ = 1 << 0;
        const MAP_WRITE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const INDEX = 1 << 4;
        const VERTEX = 1 << 5;
        const UNIFORM = 1 << 6;
        const STORAGE = 1 << 7;
        const INDIRECT = 1 << 8;
        const QUERY_RESOLVE = 1 << 9;
    }
}

#[cfg(feature = "webgpu")]
impl From<BufferUsages> for wgpu::BufferUsages {
    fn from(usage: BufferUsages) -> Self {
        wgpu::BufferUsages::from_bits_truncate(usage.bits())
    }
}

#[cfg(feature = "webgpu")]
impl From<wgpu::BufferUsages> for BufferUsages {
    fn from(usage: wgpu::BufferUsages) -> Self {
        BufferUsages::from_bits_truncate(usage.bits())
    }
}

/// Alias for [`BufferUsages`].
pub type BufferOptions = BufferUsages;

/// The type of a shader resource descriptor binding.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DescriptorType {
    /// Uniform buffer descriptor (read-only, typically smaller, faster)
    Uniform,
    /// Storage buffer descriptor (read-write capable, larger)
    Storage {
        /// If true, the buffer is read-only in the shader.
        read_only: bool,
    },
}

impl DescriptorType {
    /// Creates a read-write storage descriptor.
    pub fn storage() -> Self {
        Self::Storage { read_only: false }
    }

    /// Creates a read-only storage descriptor.
    pub fn storage_readonly() -> Self {
        Self::Storage { read_only: true }
    }
}

impl Default for DescriptorType {
    fn default() -> Self {
        Self::storage()
    }
}

/// Identifies a shader resource binding by its space (descriptor set), index, and type.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ShaderBinding {
    /// Binding space (aka. binding group).
    pub space: u32,
    /// Binding index.
    pub index: u32,
    /// Descriptor type (uniform or storage).
    pub descriptor_type: DescriptorType,
}

impl From<(u32, u32)> for ShaderBinding {
    fn from((space, index): (u32, u32)) -> Self {
        Self {
            space,
            index,
            descriptor_type: DescriptorType::default(),
        }
    }
}

/// A value that can be sent to the GPU.
///
/// # Safety
///
/// The value must comply to the safety requirements of all the backends it is implemented for.
pub unsafe trait DeviceValue: 'static + Clone + Copy + MaybeSendSync {}

// TODO: don’t do a blanket impl?
unsafe impl<T: 'static + Clone + Copy + MaybeSendSync> DeviceValue for T {}

/// Marker trait that resolves to `Send + Sync` on native targets and is
/// unconditionally implemented on wasm32 (where threads are not available).
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSendSync for T {}

/// Marker trait that resolves to `Send + Sync` on native targets and is
/// unconditionally implemented on wasm32 (where threads are not available).
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync> MaybeSendSync for T {}

/// Core abstraction over GPU compute backends (WebGPU, CUDA, CPU).
///
/// Provides the interface for loading shader modules, creating buffers,
/// encoding commands, and dispatching compute work.
pub trait Backend: 'static + Sized + MaybeSendSync {
    /// Human-readable name of this backend (e.g. `"webgpu"`, `"cuda"`).
    const NAME: &'static str;
    /// The shader compilation target this backend consumes.
    const TARGET: CompileTarget;

    /// Error type for this backend.
    type Error: Error + 'static + Send + Sync + From<ShaderArgsError>;
    /// GPU buffer type parameterized by element type.
    type Buffer<T: DeviceValue>: MaybeSendSync + Buffer<Self, T>;
    /// Immutable view into a buffer.
    type BufferSlice<'b, T: DeviceValue>;
    /// Command encoder that records GPU commands for batched submission.
    type Encoder: MaybeSendSync + Encoder<Self>;
    /// A compute pass within an encoder.
    type Pass: MaybeSendSync;
    /// Timing query handle for profiling compute passes.
    type Timestamps;
    /// A loaded shader module (WGSL, SPIR-V, or PTX).
    type Module;
    /// A compiled compute function (pipeline) extracted from a module.
    type Function: MaybeSendSync;
    /// An in-progress dispatch that collects bindings before launch.
    type Dispatch<'a>: Dispatch<'a, Self>
    where
        Self: 'a;

    /// Downcasts to the WebGPU backend, if applicable.
    #[cfg(feature = "webgpu")]
    fn as_webgpu(&self) -> Option<&WebGpu> {
        None
    }

    /// Downcasts to the CUDA backend, if applicable.
    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&Cuda> {
        None
    }

    /*
     * Module/function loading.
     */
    /// Loads a shader module from a string (WGSL text or PTX source).
    fn load_module(&self, data: &str) -> Result<Self::Module, Self::Error> {
        self.load_module_bytes(data.as_bytes())
    }
    /// Loads a shader module from raw bytes (SPIR-V binary, WGSL text, or PTX source).
    fn load_module_bytes(&self, data: &[u8]) -> Result<Self::Module, Self::Error>;
    /// Load a function from a module.
    ///
    /// The `push_constant_size` parameter specifies the size of push constants in bytes.
    /// Set to 0 if the shader doesn't use push constants.
    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
    ) -> Result<Self::Function, Self::Error>;

    /// Load a function from a module with explicit bind group layout information.
    ///
    /// This allows creating explicit bind group layouts from ShaderArgs, rather than
    /// relying on the backend to auto-generate them from shader reflection.
    ///
    /// The default implementation ignores the layout info and calls `load_function`.
    fn load_function_with_layouts(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
        _layouts: &BindGroupLayoutInfo,
    ) -> Result<Self::Function, Self::Error> {
        self.load_function(module, entry_point, push_constant_size)
    }

    /*
     * Kernel dispatch.
     */
    /// Creates a new command encoder for recording GPU operations.
    fn begin_encoding(&self) -> Self::Encoder;
    /// Begins a compute dispatch within a pass, binding the given function.
    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a>;
    /// Blocks until all submitted GPU work has completed.
    fn synchronize(&self) -> Result<(), Self::Error>;
    /// Submits the recorded commands in the encoder for execution.
    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error>;

    /*
     * Buffer handling.
     */
    /// Creates a GPU buffer initialized with `data`.
    fn init_buffer<T: DeviceValue + NoUninit>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;

    // fn init_buffer_bytes<T: Copy>(&self, bytes: &[u8], usage: BufferUsages) -> Result<Self::Buffer<T>, Self::Error>;

    /// Creates a zero-initialized GPU buffer of `len` elements.
    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;

    /// Writes `data` into `buffer` starting at the given element `offset`.
    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error>;
    /// Reads `buffer` contents into `data`. Requires the buffer to be mappable.
    fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync;
    /// Slower version of `read_buffer` that doesn’t require `buffer` to be a mapped staging
    /// buffer.
    ///
    /// This is slower, but more convenient than [`Self::read_buffer`] because it takes care of
    /// creating a staging buffer, running a buffer-to-buffer copy from `buffer` to the staging
    /// buffer, and running a buffer-to-host copy from the staging buffer to `data`.
    fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync;

    /// Convenience wrapper around [`slow_read_buffer`](Self::slow_read_buffer) that allocates
    /// and returns a `Vec<T>`.
    fn slow_read_vec<T: MaybeSendSync + DeviceValue + AnyBitPattern + Default>(
        &self,
        buffer: &Self::Buffer<T>,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>> + MaybeSendSync {
        async move {
            let mut result = vec![T::default(); buffer.len()];
            self.slow_read_buffer(buffer, &mut result).await?;
            Ok(result)
        }
    }
}

/// Records GPU commands (compute passes, buffer copies) for batched submission.
pub trait Encoder<B: Backend> {
    /// Begins a labeled compute pass, optionally recording timestamps for profiling.
    fn begin_pass(&mut self, label: &str, timestamps: Option<&mut B::Timestamps>) -> B::Pass;
    /// Copies `copy_len` elements from `source` to `target` at the given offsets.
    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &B::Buffer<T>,
        source_offset: usize,
        target: &mut B::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), B::Error>;
}

/// An in-progress compute dispatch that collects bindings and launches kernels.
pub trait Dispatch<'a, B: Backend> {
    /// Sets push constants for this dispatch.
    ///
    /// Push constants are small pieces of data that can be updated frequently
    /// and are passed directly to the shader without buffer indirection.
    /// This is only available when the `push_constants` feature is enabled.
    #[cfg(feature = "push_constants")]
    fn set_push_constants(&mut self, data: &[u8]);

    /// Launches the compute kernel with the given dispatch grid and workgroup size.
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, B>>,
        workgroups: [u32; 3],
    ) -> Result<(), B::Error>;
}

/// A GPU buffer holding elements of type `T`.
pub trait Buffer<B: Backend, T: DeviceValue> {
    /// Returns `true` if the buffer contains no elements.
    fn is_empty(&self) -> bool;
    /// Returns the number of elements in this buffer.
    fn len(&self) -> usize
    where
        T: Sized;
    /// Returns an immutable slice view of the entire buffer.
    fn as_slice(&self) -> B::BufferSlice<'_, T> {
        self.slice(..)
    }
    /// Returns an immutable slice view of the given element range.
    fn slice(&self, range: impl RangeBounds<usize>) -> B::BufferSlice<'_, T>;
    /// Returns the usage flags this buffer was created with.
    fn usage(&self) -> BufferUsages;
}

/// Specifies how to dispatch a compute kernel.
pub enum DispatchGrid<'a, B: Backend> {
    /// Dispatch with an explicit workgroup grid (workgroup counts).
    Grid([u32; 3]),
    /// Dispatch with a thread count (will be divided by workgroup size).
    ThreadCount([u32; 3]),
    /// Indirect dispatch from a GPU buffer containing workgroup counts.
    Indirect(&'a B::Buffer<[u32; 3]>),
}

impl<'a, B: Backend> DispatchGrid<'a, B> {
    /// Resolves `ThreadCount` to `Grid` by dividing by the workgroup size.
    /// Other variants are returned as-is.
    pub fn resolve(self, workgroup_size: [u32; 3]) -> Self {
        match self {
            DispatchGrid::ThreadCount(threads) => DispatchGrid::Grid([
                threads[0].div_ceil(workgroup_size[0]),
                threads[1].div_ceil(workgroup_size[1]),
                threads[2].div_ceil(workgroup_size[2]),
            ]),
            other => other,
        }
    }
}

impl<'a, B: Backend> From<u32> for DispatchGrid<'a, B> {
    fn from(num_threads: u32) -> DispatchGrid<'a, B> {
        DispatchGrid::ThreadCount([num_threads, 1, 1])
    }
}

impl<'a, B: Backend> From<usize> for DispatchGrid<'a, B> {
    fn from(num_threads: usize) -> DispatchGrid<'a, B> {
        DispatchGrid::ThreadCount([num_threads as u32, 1, 1])
    }
}

impl<'a, B: Backend> From<i32> for DispatchGrid<'a, B> {
    fn from(num_threads: i32) -> DispatchGrid<'a, B> {
        assert!(num_threads >= 0);
        DispatchGrid::ThreadCount([num_threads as u32, 1, 1])
    }
}

impl<'a, B: Backend> From<[u32; 3]> for DispatchGrid<'a, B> {
    fn from(num_threads: [u32; 3]) -> DispatchGrid<'a, B> {
        DispatchGrid::ThreadCount(num_threads)
    }
}

impl<'a, B: Backend> From<[usize; 3]> for DispatchGrid<'a, B> {
    fn from(num_threads: [usize; 3]) -> DispatchGrid<'a, B> {
        DispatchGrid::ThreadCount(num_threads.map(|num_threads| num_threads as u32))
    }
}

impl<'a, B: Backend> From<[i32; 3]> for DispatchGrid<'a, B> {
    fn from(num_threads: [i32; 3]) -> DispatchGrid<'a, B> {
        DispatchGrid::ThreadCount(num_threads.map(|num_threads| {
            assert!(num_threads >= 0);
            num_threads as u32
        }))
    }
}
