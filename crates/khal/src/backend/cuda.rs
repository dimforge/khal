use crate::backend::{
    Backend, BufferUsages, CompileTarget, DeviceValue, Dispatch, DispatchGrid, Encoder,
    MaybeSendSync, ShaderBinding,
};
use crate::shader::{BindGroupLayoutInfo, ShaderArgsError};
use bytemuck::{AnyBitPattern, NoUninit};
use cudarc::driver::{self, CudaContext, CudaStream};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};

// ── Core backend ───────────────────────────────────────────────────────

/// CUDA backend for running compute shaders via NVIDIA's CUDA driver API.
#[derive(Clone)]
pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Cache of loaded PTX modules keyed by content hash.
    /// Avoids re-loading the same PTX for every kernel in a crate.
    module_cache: Arc<Mutex<HashMap<u64, Arc<driver::CudaModule>>>>,
}

impl Cuda {
    /// Creates a new CUDA backend using the specified device ordinal.
    pub fn new(device_ordinal: usize) -> Result<Self, CudaBackendError> {
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            module_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Returns the underlying cudarc context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the default stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// ── Error ──────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum CudaBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error("CUDA driver error: {0}")]
    Driver(#[from] driver::DriverError),
    #[error("Invalid PTX module")]
    InvalidPtx,
}

// ── Buffer ─────────────────────────────────────────────────────────────

/// A CUDA device buffer with element count and usage metadata.
pub struct CudaBuffer<T: DeviceValue> {
    /// Device memory allocation. `None` for zero-length buffers.
    inner: Option<driver::CudaSlice<u8>>,
    /// Cached raw device pointer (CUdeviceptr). Set at creation time.
    raw_ptr: u64,
    len: usize,
    usage: BufferUsages,
    _marker: PhantomData<T>,
}

impl<T: DeviceValue> CudaBuffer<T> {
    /// Returns the raw device pointer as a `u64`, or 0 for empty buffers.
    pub fn device_ptr_raw(&self) -> u64 {
        self.raw_ptr
    }

    /// Returns the total byte length of this buffer.
    pub fn byte_len(&self) -> u64 {
        (self.len * std::mem::size_of::<T>()) as u64
    }

    /// Returns a reference to the inner CudaSlice, if any.
    pub fn inner(&self) -> Option<&driver::CudaSlice<u8>> {
        self.inner.as_ref()
    }

    /// Returns a mutable reference to the inner CudaSlice, if any.
    pub fn inner_mut(&mut self) -> Option<&mut driver::CudaSlice<u8>> {
        self.inner.as_mut()
    }
}

// ── Buffer slice ───────────────────────────────────────────────────────

/// An immutable view into a CUDA device buffer.
#[derive(Clone, Copy)]
pub struct CudaBufferSlice {
    pub(crate) device_ptr: u64,
    pub(crate) byte_offset: u64,
    pub(crate) byte_len: u64,
}

impl CudaBufferSlice {
    /// Returns the device pointer offset to this slice's start.
    pub fn offset_ptr(&self) -> u64 {
        self.device_ptr + self.byte_offset
    }
}

// ── Module / Function ──────────────────────────────────────────────────

/// A loaded PTX module on the CUDA device.
#[derive(Clone)]
pub struct CudaModule {
    pub(crate) inner: Arc<driver::CudaModule>,
}

/// A CUDA kernel function extracted from a loaded module.
#[derive(Clone)]
pub struct CudaFunction {
    pub(crate) func: driver::CudaFunction,
}

// ── Encoder / Pass ─────────────────────────────────────────────────────

/// CUDA command encoder. CUDA doesn't batch commands like WebGPU, so this
/// is essentially a thin wrapper that holds a reference to the stream.
pub struct CudaEncoder {
    pub(crate) stream: Arc<CudaStream>,
}

/// CUDA compute pass. CUDA doesn't have explicit compute pass boundaries.
pub struct CudaPass {
    pub(crate) stream: Arc<CudaStream>,
    /// Optional timing: records an end event on drop and pushes to the shared pending list.
    timing: Option<CudaPassTiming>,
}

impl Drop for CudaPass {
    fn drop(&mut self) {
        if let Some(mut timing) = self.timing.take() {
            if let Some(start) = timing.start.take() {
                if let Ok(end) = timing
                    .stream
                    .record_event(Some(driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                {
                    timing.pending.lock().unwrap().push(CudaPendingTimestamp {
                        label: std::mem::take(&mut timing.label),
                        start,
                        end,
                    });
                }
            }
        }
    }
}

struct CudaPassTiming {
    label: String,
    start: Option<driver::CudaEvent>,
    stream: Arc<CudaStream>,
    pending: Arc<Mutex<Vec<CudaPendingTimestamp>>>,
}

// ── Dispatch ───────────────────────────────────────────────────────────

/// Collects kernel arguments and launches a CUDA kernel.
pub struct CudaDispatch<'a> {
    pub(crate) stream: &'a Arc<CudaStream>,
    pub(crate) function: &'a CudaFunction,
    /// Collected arguments: (binding, device_ptr, byte_len).
    pub(crate) args: Vec<(ShaderBinding, u64, u64)>,
    #[cfg(feature = "push_constants")]
    pub(crate) push_constants: Vec<u8>,
}

impl<'a> CudaDispatch<'a> {
    /// Adds a buffer argument at the given binding location.
    pub fn set_arg(&mut self, binding: ShaderBinding, device_ptr: u64, byte_len: u64) {
        self.args.push((binding, device_ptr, byte_len));
    }

    /// Sets push constant data for this dispatch.
    #[cfg(feature = "push_constants")]
    pub fn set_push_constants(&mut self, data: &[u8]) {
        self.push_constants.clear();
        self.push_constants.extend_from_slice(data);
    }
}

// ── Timestamps ─────────────────────────────────────────────────────────

/// A pending timestamp pair waiting for GPU completion.
struct CudaPendingTimestamp {
    label: String,
    start: driver::CudaEvent,
    end: driver::CudaEvent,
}

/// CUDA event-based timing for compute passes.
pub struct CudaTimestamps {
    stream: Arc<CudaStream>,
    pending: Arc<Mutex<Vec<CudaPendingTimestamp>>>,
}

impl CudaTimestamps {
    /// Creates a new timestamp manager using the CUDA backend's stream.
    pub fn new(cuda: &Cuda) -> Self {
        Self {
            stream: cuda.stream.clone(),
            pending: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Clears all pending timestamps.
    pub fn reset(&mut self) {
        self.pending.lock().unwrap().clear();
    }

    /// Reads timestamp results. Must be called after stream synchronization.
    pub fn read(&self) -> Result<Vec<super::GpuTimestamp>, CudaBackendError> {
        let pending = self.pending.lock().unwrap();
        let mut entries = Vec::with_capacity(pending.len());
        for p in pending.iter() {
            let elapsed_ms =
                unsafe { driver::result::event::elapsed(p.start.cu_event(), p.end.cu_event()) }?;
            entries.push(super::GpuTimestamp {
                label: p.label.clone(),
                duration_ms: elapsed_ms as f64,
            });
        }
        Ok(entries)
    }
}

// ── Backend trait impl ─────────────────────────────────────────────────

impl Backend for Cuda {
    const NAME: &'static str = "cuda";
    const TARGET: CompileTarget = CompileTarget::Ptx;

    type Error = CudaBackendError;
    type Buffer<T: DeviceValue> = CudaBuffer<T>;
    type BufferSlice<'b, T: DeviceValue> = CudaBufferSlice;
    type Encoder = CudaEncoder;
    type Pass = CudaPass;
    type Timestamps = CudaTimestamps;
    type Module = CudaModule;
    type Function = CudaFunction;
    type Dispatch<'a> = CudaDispatch<'a>;

    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&Cuda> {
        Some(self)
    }

    /*
     * Module / function loading.
     */
    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        // Check the module cache first to avoid re-loading the same PTX.
        let hash = fxhash(bytes);
        {
            let cache = self.module_cache.lock().unwrap();
            if let Some(module) = cache.get(&hash) {
                return Ok(CudaModule {
                    inner: module.clone(),
                });
            }
        }

        // Expect PTX text bytes.
        let ptx_str = std::str::from_utf8(bytes).map_err(|_| CudaBackendError::InvalidPtx)?;
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_str.to_string());
        let module = self.ctx.load_module(ptx)?;

        // Cache the loaded module.
        self.module_cache
            .lock()
            .unwrap()
            .insert(hash, module.clone());

        Ok(CudaModule { inner: module })
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
        _push_constant_size: u32,
    ) -> Result<Self::Function, Self::Error> {
        let func = module.inner.load_function(entry_point)?;
        Ok(CudaFunction { func })
    }

    fn load_function_with_layouts(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
        _layouts: &BindGroupLayoutInfo,
    ) -> Result<Self::Function, Self::Error> {
        // CUDA doesn't use bind group layouts.
        self.load_function(module, entry_point, push_constant_size)
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        CudaEncoder {
            stream: self.stream.clone(),
        }
    }

    fn begin_dispatch<'a>(
        &'a self,
        _pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a> {
        CudaDispatch {
            stream: &self.stream,
            function,
            args: Vec::new(),
            #[cfg(feature = "push_constants")]
            push_constants: Vec::new(),
        }
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn submit(&self, _encoder: Self::Encoder) -> Result<(), Self::Error> {
        // CUDA operations are submitted immediately; nothing to flush.
        Ok(())
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + NoUninit>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if data.is_empty() {
            return Ok(CudaBuffer {
                inner: None,
                raw_ptr: 0,
                len: 0,
                usage,
                _marker: PhantomData,
            });
        }
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let slice = self.stream.clone_htod(bytes)?;
        let raw_ptr = extract_raw_ptr(&slice, &self.stream);
        Ok(CudaBuffer {
            inner: Some(slice),
            raw_ptr,
            len: data.len(),
            usage,
            _marker: PhantomData,
        })
    }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if len == 0 {
            return Ok(CudaBuffer {
                inner: None,
                raw_ptr: 0,
                len: 0,
                usage,
                _marker: PhantomData,
            });
        }
        let byte_len = len * std::mem::size_of::<T>();
        let slice = self.stream.alloc_zeros::<u8>(byte_len)?;
        let raw_ptr = extract_raw_ptr(&slice, &self.stream);
        Ok(CudaBuffer {
            inner: Some(slice),
            raw_ptr,
            len,
            usage,
            _marker: PhantomData,
        })
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        if let Some(ref mut inner) = buffer.inner {
            let byte_offset = offset as usize * std::mem::size_of::<T>();
            let bytes: &[u8] = bytemuck::cast_slice(data);
            let mut dst_view = inner.slice_mut(byte_offset..byte_offset + bytes.len());
            self.stream.memcpy_htod(bytes, &mut dst_view)?;
        }
        Ok(())
    }

    fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move {
            if let Some(ref inner) = buffer.inner {
                let bytes: Vec<u8> = self.stream.clone_dtoh(inner)?;
                let copy_len = bytes.len().min(std::mem::size_of_val(out));
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr() as *mut u8,
                        copy_len,
                    );
                }
            }
            Ok(())
        }
    }

    fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        // CUDA doesn't need staging buffers.
        self.read_buffer(buffer, out)
    }
}

// ── Encoder ────────────────────────────────────────────────────────────

impl Encoder<Cuda> for CudaEncoder {
    fn begin_pass(&mut self, label: &str, timestamps: Option<&mut CudaTimestamps>) -> CudaPass {
        let timing = timestamps.and_then(|ts| {
            let start = ts
                .stream
                .record_event(Some(driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .ok()?;
            Some(CudaPassTiming {
                label: label.to_string(),
                start: Some(start),
                stream: ts.stream.clone(),
                pending: ts.pending.clone(),
            })
        });
        CudaPass {
            stream: self.stream.clone(),
            timing,
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &CudaBuffer<T>,
        source_offset: usize,
        target: &mut CudaBuffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), CudaBackendError> {
        if let (Some(src), Some(dst)) = (&source.inner, &mut target.inner) {
            let elem_size = std::mem::size_of::<T>();
            let src_byte_offset = source_offset * elem_size;
            let dst_byte_offset = target_offset * elem_size;
            let byte_len = copy_len * elem_size;
            let src_view = src.slice(src_byte_offset..src_byte_offset + byte_len);
            let mut dst_view = dst.slice_mut(dst_byte_offset..dst_byte_offset + byte_len);
            self.stream.memcpy_dtod(&src_view, &mut dst_view)?;
        }
        Ok(())
    }
}

// ── Dispatch ───────────────────────────────────────────────────────────

impl<'a> Dispatch<'a, Cuda> for CudaDispatch<'a> {
    #[cfg(feature = "push_constants")]
    fn set_push_constants(&mut self, data: &[u8]) {
        self.push_constants.clear();
        self.push_constants.extend_from_slice(data);
    }

    fn launch<'b>(
        mut self,
        grid: impl Into<DispatchGrid<'b, Cuda>>,
        block_dim: [u32; 3],
    ) -> Result<(), CudaBackendError> {
        let grid_dim = match grid.into() {
            DispatchGrid::Grid(g) => g,
            DispatchGrid::ThreadCount(t) => [
                t[0].div_ceil(block_dim[0]),
                t[1].div_ceil(block_dim[1]),
                t[2].div_ceil(block_dim[2]),
            ],
            DispatchGrid::Indirect(buffer) => {
                // CUDA doesn't support indirect dispatch natively.
                // Read the 12-byte dispatch args from device memory.
                self.stream.synchronize()?;
                if let Some(ref inner) = buffer.inner {
                    let bytes: Vec<u8> = self.stream.clone_dtoh(inner)?;
                    if bytes.len() >= 12 {
                        [
                            u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                            u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                            u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                        ]
                    } else {
                        return Ok(());
                    }
                } else {
                    return Ok(());
                }
            }
        };

        // Skip zero-work dispatches.
        if grid_dim[0] * grid_dim[1] * grid_dim[2] == 0 {
            return Ok(());
        }

        // Sort args by (space, index) to match the kernel parameter order.
        self.args
            .sort_by_key(|(binding, _, _)| (binding.space, binding.index));

        // Build the kernel parameter list.
        // Each storage buffer becomes (device_ptr: u64, byte_len: u64).
        // Uniform buffers only pass the pointer.
        let mut param_values: Vec<u64> = Vec::with_capacity(self.args.len() * 2);
        for (binding, device_ptr, byte_len) in &self.args {
            param_values.push(*device_ptr);
            match binding.descriptor_type {
                crate::backend::DescriptorType::Storage { .. } => {
                    param_values.push(*byte_len);
                }
                crate::backend::DescriptorType::Uniform => {
                    // Uniform buffers only pass the pointer.
                }
            }
        }

        // Append push constants as raw bytes (padded to u64 alignment).
        #[cfg(feature = "push_constants")]
        {
            let pc = &self.push_constants;
            let mut offset = 0;
            while offset + 8 <= pc.len() {
                let val = u64::from_ne_bytes(pc[offset..offset + 8].try_into().unwrap());
                param_values.push(val);
                offset += 8;
            }
            if offset < pc.len() {
                let mut buf = [0u8; 8];
                buf[..pc.len() - offset].copy_from_slice(&pc[offset..]);
                param_values.push(u64::from_ne_bytes(buf));
            }
        }

        // Launch using cudarc's launch_builder API.
        let mut builder = self.stream.launch_builder(&self.function.func);
        for value in &param_values {
            use cudarc::driver::PushKernelArg;
            builder.arg(value);
        }

        let cfg = driver::LaunchConfig {
            grid_dim: (grid_dim[0], grid_dim[1], grid_dim[2]),
            block_dim: (block_dim[0], block_dim[1], block_dim[2]),
            shared_mem_bytes: 0,
        };

        unsafe {
            builder.launch(cfg)?;
        }

        Ok(())
    }
}

// ── Buffer trait impl ──────────────────────────────────────────────────

impl<T: DeviceValue> crate::backend::Buffer<Cuda, T> for CudaBuffer<T> {
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        self.len
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> CudaBufferSlice {
        let elem_size = std::mem::size_of::<T>() as u64;
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n as u64 * elem_size,
            std::ops::Bound::Excluded(&n) => (n as u64 + 1) * elem_size,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => (n as u64 + 1) * elem_size,
            std::ops::Bound::Excluded(&n) => n as u64 * elem_size,
            std::ops::Bound::Unbounded => self.byte_len(),
        };
        CudaBufferSlice {
            device_ptr: self.device_ptr_raw(),
            byte_offset: start,
            byte_len: end - start,
        }
    }

    fn usage(&self) -> BufferUsages {
        self.usage
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Extracts the raw CUdeviceptr from a CudaSlice using the DevicePtr trait.
fn extract_raw_ptr<T>(slice: &driver::CudaSlice<T>, stream: &CudaStream) -> u64 {
    use cudarc::driver::DevicePtr;
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}

/// Simple FNV-1a hash for module cache keys.
fn fxhash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
