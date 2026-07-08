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
use std::sync::{Arc, Mutex, OnceLock};

static KERNEL_PROFILE: OnceLock<Mutex<HashMap<String, (u64, u128)>>> = OnceLock::new();
fn kernel_profile() -> &'static Mutex<HashMap<String, (u64, u128)>> {
    KERNEL_PROFILE.get_or_init(|| Mutex::new(HashMap::new()))
}
/// Print accumulated per-kernel timings (sorted desc) and clear. No-op if unused.
pub fn dump_kernel_profile() {
    let mut map = kernel_profile().lock().unwrap();
    if map.is_empty() {
        return;
    }
    let mut rows: Vec<_> = map.iter().map(|(k, (c, ns))| (k.clone(), *c, *ns)).collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    let total: u128 = rows.iter().map(|r| r.2).sum();
    eprintln!(
        "\n=== KHAL_CUDA_PROFILE: {} kernels, {:.3} ms total (serialized) ===",
        rows.len(),
        total as f64 / 1e6
    );
    for (n, c, ns) in &rows {
        eprintln!(
            "{:>9.3} ms  {:>6}x  {:>9.2} us  {:>6.2}%  {}",
            *ns as f64 / 1e6,
            c,
            (*ns as f64 / *c as f64) / 1e3,
            *ns as f64 / total as f64 * 100.0,
            n
        );
    }
    map.clear();
}

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

    /// Compute capability `(major, minor)` of the underlying device, e.g.
    /// `(12, 0)` for Blackwell `sm_120`. Used to pick the default backend.
    pub fn compute_capability(&self) -> Result<(i32, i32), CudaBackendError> {
        Ok(self.ctx.compute_capability()?)
    }

    /// Returns the default stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Begin CUDA stream capture on the default stream: subsequent kernel
    /// launches are *recorded* into a graph instead of executed, until
    /// [`Cuda::end_capture`]. Used by the GPU-resident rollout to capture a
    /// repeated dispatch sequence (e.g. the physics decimation loop) once and
    /// replay it with a single launch — eliminating per-launch host encode/submit
    /// overhead. THREAD_LOCAL mode scopes capture to the current thread.
    ///
    /// The captured sequence must be replay-safe: no allocation/free, no host
    /// syncs, and stable buffer addresses across replays.
    pub fn begin_capture(&self) -> Result<(), CudaBackendError> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        Ok(())
    }

    /// End stream capture and instantiate the recorded graph for replay. Errors
    /// if nothing was captured (the stream wasn't in capture, or recorded no work).
    pub fn end_capture(&self) -> Result<CapturedGraph, CudaBackendError> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        // No instantiate flags (0). The flags enum has no zero variant; it is
        // consumed as `flags as u32 as u64`, so a 0-valued enum means "no flags".
        let no_flags: CUgraphInstantiate_flags = unsafe { std::mem::transmute(0u32) };
        let graph = self
            .stream
            .end_capture(no_flags)?
            .ok_or(CudaBackendError::CaptureFailed)?;
        Ok(CapturedGraph { graph })
    }
}

/// A captured, instantiated CUDA graph. Replay the whole recorded kernel
/// sequence with a single [`CapturedGraph::launch`] (one `cuGraphLaunch`),
/// instead of re-encoding/re-submitting each dispatch from the host.
pub struct CapturedGraph {
    graph: driver::CudaGraph,
}

impl CapturedGraph {
    /// Replay the captured kernel sequence with a single graph launch.
    pub fn launch(&self) -> Result<(), CudaBackendError> {
        self.graph.launch()?;
        Ok(())
    }

    /// Pre-upload the graph's resources so the first [`CapturedGraph::launch`]
    /// doesn't pay instantiation/upload cost.
    pub fn upload(&self) -> Result<(), CudaBackendError> {
        self.graph.upload()?;
        Ok(())
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
    #[error("CUDA stream capture produced no graph")]
    CaptureFailed,
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
    pub(crate) name: String,
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

    /// Whether no recorded-but-untaken timestamps remain (safe to record a new
    /// frame). False while events from a previous frame await [`try_take`].
    pub fn is_idle(&self) -> bool {
        self.pending.lock().unwrap().is_empty()
    }

    /// Initiates a non-blocking readback. No-op for CUDA: completion is probed
    /// directly via the recorded events in [`try_take`](Self::try_take).
    pub fn request_read(&mut self) {}

    /// Non-blocking poll of the recorded timestamps.
    ///
    /// Returns `Some(results)` once every event has completed, or `None` while
    /// any is still pending. `event::elapsed` reports `CUDA_ERROR_NOT_READY`
    /// (an `Err`) until both events of a pair have been recorded by the GPU, so
    /// a successful read doubles as the readiness probe.
    pub fn try_take(&mut self) -> Option<Vec<super::GpuTimestamp>> {
        self.read().ok()
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

        // Accept either PTX text or a pre-linked CUBIN (detected by ELF magic).
        // A cubin is required when the module references symbols the driver JIT
        // cannot resolve on its own (e.g. libdevice `__nv_*` math), which a
        // toolchain links into a self-contained binary ahead of time.
        let ptx = if bytes.starts_with(&[0x7f, b'E', b'L', b'F']) {
            cudarc::nvrtc::Ptx::from_binary(bytes.to_vec())
        } else {
            let ptx_str = std::str::from_utf8(bytes).map_err(|_| CudaBackendError::InvalidPtx)?;
            cudarc::nvrtc::Ptx::from_src(ptx_str.to_string())
        };
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
        let func = match module.inner.load_function(entry_point) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[khal-cuda load_function FAIL] {} -> {:?}", entry_point, e);
                return Err(e.into());
            }
        };
        Ok(CudaFunction { func, name: entry_point.to_string() })
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

        let trace = std::env::var_os("KHAL_CUDA_TRACE").is_some();
        if trace {
            eprintln!(
                "[khal-cuda launch] {} nargs={} grid={:?} block={:?}",
                self.function.name, param_values.len(), grid_dim, block_dim
            );
        }
        let prof_start = if std::env::var_os("KHAL_CUDA_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        unsafe {
            builder.launch(cfg)?;
        }
        if let Some(t0) = prof_start {
            self.stream.synchronize()?;
            let ns = t0.elapsed().as_nanos();
            let mut map = kernel_profile().lock().unwrap();
            let e = map.entry(self.function.name.clone()).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
        if trace {
            match self.stream.synchronize() {
                Ok(()) => eprintln!("[khal-cuda   ok  ] {}", self.function.name),
                Err(e) => {
                    eprintln!("[khal-cuda  FAIL ] {} -> {:?}", self.function.name, e);
                    return Err(e.into());
                }
            }
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
