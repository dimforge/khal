//! Native Metal compute backend.
//!
//! Translates SPIR-V to MSL via `naga` at function-load time, then uses
//! Apple's Metal API directly through the [`metal`] crate. Unlike the
//! WebGPU backend (which goes through `wgpu`), this backend tracks no
//! per-resource state and inserts no implicit barriers between dispatches —
//! callers must use [`crate::backend::Encoder::memory_barrier`] when a
//! later dispatch reads from a buffer written by an earlier one in the
//! same compute pass.

use crate::backend::{
    Backend, BufferUsages, CompileTarget, DescriptorType, DeviceValue, Dispatch, DispatchGrid,
    Encoder, GpuTimestamp, MaybeSendSync, ShaderBinding,
};
use crate::shader::{BindGroupLayoutInfo, ShaderArgsError};
use bytemuck::{AnyBitPattern, NoUninit};
// metal re-exports objc; pull in its macros so msg_send! / sel! resolve.
use block::ConcreteBlock;
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer as MtlBuffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePassDescriptor,
    ComputePipelineDescriptor, ComputePipelineState, CounterSampleBuffer,
    CounterSampleBufferDescriptor, CounterSet, Device, Library, MTLCounterSamplingPoint,
    MTLDispatchType, MTLResourceOptions, MTLSize, MTLStorageMode, NSRange, NSUInteger,
};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ── Core backend ───────────────────────────────────────────────────────

/// Native Metal backend. Wraps an [`MTLDevice`](metal::Device) and queue.
#[derive(Clone)]
pub struct Metal {
    device: Device,
    queue: CommandQueue,
    /// Cache of compiled MSL libraries keyed by SPIR-V content hash.
    module_cache: Arc<Mutex<HashMap<u64, MetalModule>>>,
    /// Capabilities for GPU timestamp queries; `None` if the device or
    /// driver doesn't expose stage-boundary timestamp sampling.
    timing_caps: Option<Arc<MetalTimingCaps>>,
}

// SAFETY: metal::Device and CommandQueue are thread-safe (MTLDevice/MTLCommandQueue
// are documented as Send+Sync by Apple). The metal-rs crate doesn't auto-derive
// these, but the underlying Objective-C objects are.
unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}

impl Metal {
    /// Commits an empty command buffer with a completion handler that flips the
    /// returned flag once the GPU finishes it.
    ///
    /// Because the queue completes command buffers in commit order, the flag
    /// being set means every earlier submission has finished. The handler fires
    /// asynchronously on the GPU's completion — unlike polling `status()`, this
    /// needs no `synchronize()`/drain to make progress, so the non-blocking
    /// readback paths ([`MetalTimestamps`], `GpuReadback`) work on their own.
    pub(crate) fn commit_completion(&self) -> Arc<AtomicBool> {
        let done = Arc::new(AtomicBool::new(false));
        let done_handler = done.clone();
        let cb = self.queue.new_command_buffer();
        let handler = ConcreteBlock::new(move |_cb: &metal::CommandBufferRef| {
            done_handler.store(true, Ordering::Release);
        })
        .copy();
        cb.add_completed_handler(&handler);
        cb.commit();
        done
    }
}

/// Cached info needed to issue GPU timestamp queries on this device.
struct MetalTimingCaps {
    /// The device's "timestamp" common counter set.
    counter_set: CounterSet,
    /// Multiplier converting raw timestamp ticks to nanoseconds.
    /// `1.0` on Apple Silicon and AMD; `83.333…` on older Intel iGPUs.
    period_ns: f64,
}

// SAFETY: CounterSet wraps an MTLCounterSet which is thread-safe.
unsafe impl Send for MetalTimingCaps {}
unsafe impl Sync for MetalTimingCaps {}

impl Metal {
    /// Creates a new Metal backend using the system default device.
    pub fn new() -> Result<Self, MetalBackendError> {
        let device = Device::system_default().ok_or(MetalBackendError::NoDevice)?;
        let queue = device.new_command_queue();
        let timing_caps = detect_timing_caps(&device);
        Ok(Self {
            device,
            queue,
            module_cache: Arc::new(Mutex::new(HashMap::new())),
            timing_caps,
        })
    }

    /// Returns the underlying Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the command queue used by this backend.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Whether GPU timestamp queries are supported on this device.
    pub fn timestamp_supported(&self) -> bool {
        self.timing_caps.is_some()
    }
}

/// Probes the device for timestamp counter support. Returns `None` if either
/// stage-boundary sampling isn't supported or no "timestamp" counter set
/// exists (e.g. on older drivers or virtualized devices).
fn detect_timing_caps(device: &Device) -> Option<Arc<MetalTimingCaps>> {
    if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    let counter_set = device
        .counter_sets()
        .into_iter()
        .find(|cs| cs.name() == "timestamp")?;
    // Match the heuristic wgpu-hal uses: pre-Apple-Silicon Intel iGPUs report
    // timestamps in ~83.333 ns units; everything else reports nanoseconds.
    let period_ns = if device.name().starts_with("Intel") {
        83.333
    } else {
        1.0
    };
    Some(Arc::new(MetalTimingCaps {
        counter_set,
        period_ns,
    }))
}

// ── Error ──────────────────────────────────────────────────────────────

/// Errors specific to the Metal backend.
#[derive(thiserror::Error, Debug)]
pub enum MetalBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error("No Metal device available")]
    NoDevice,
    #[error("Failed to parse SPIR-V: {0}")]
    SpirVParse(String),
    #[error("Naga validation failed: {0}")]
    NagaValidation(String),
    #[error("Failed to write MSL: {0}")]
    MslWrite(String),
    #[error("Metal library compilation failed: {0}")]
    LibraryCompile(String),
    #[error("Metal pipeline creation failed: {0}")]
    PipelineCreate(String),
    #[error("Entry point `{0}` not found in module")]
    EntryPointNotFound(String),
}

// ── Buffer ─────────────────────────────────────────────────────────────

/// A Metal device buffer with element count and usage metadata.
pub struct MetalBuffer<T: DeviceValue> {
    inner: MtlBuffer,
    len: usize,
    byte_len: usize,
    usage: BufferUsages,
    _marker: PhantomData<T>,
}

// SAFETY: MTLBuffer is documented thread-safe by Apple.
unsafe impl<T: DeviceValue> Send for MetalBuffer<T> {}
unsafe impl<T: DeviceValue> Sync for MetalBuffer<T> {}

impl<T: DeviceValue> MetalBuffer<T> {
    /// Returns the underlying Metal buffer.
    pub fn raw(&self) -> &MtlBuffer {
        &self.inner
    }

    /// Returns the total byte length of this buffer.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

// ── Buffer slice ───────────────────────────────────────────────────────

/// An immutable view into a Metal device buffer.
#[derive(Clone, Copy)]
pub struct MetalBufferSlice<'a> {
    pub(crate) buffer: &'a MtlBuffer,
    pub(crate) byte_offset: u64,
    pub(crate) byte_len: u64,
}

impl<'a> MetalBufferSlice<'a> {
    /// Returns the underlying Metal buffer.
    pub fn buffer(&self) -> &'a MtlBuffer {
        self.buffer
    }

    /// Byte offset into the underlying buffer.
    pub fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Byte length of this slice.
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

// ── Module / Function ──────────────────────────────────────────────────

/// A loaded shader module: parsed naga IR ready to be translated to MSL
/// per-entry-point at function load time.
#[derive(Clone)]
pub struct MetalModule {
    naga: Arc<naga::Module>,
    info: Arc<naga::valid::ModuleInfo>,
    /// Type layouts (size, alignment) for every type in `naga.types`.
    /// Used to compute threadgroup memory allocations.
    layouter: Arc<naga::proc::Layouter>,
}

/// A Metal compute pipeline plus the (group, binding) → MSL buffer slot map.
#[derive(Clone)]
pub struct MetalFunction {
    pub(crate) pipeline: ComputePipelineState,
    /// Sorted bindings (group, binding, descriptor_type) → MSL buffer slot.
    pub(crate) slot_map: Arc<Vec<(u32, u32, u64)>>, // (group, binding, slot)
    /// MSL buffer slot for push constants, if any.
    #[cfg(feature = "push_constants")]
    pub(crate) push_constant_slot: Option<u64>,
    /// MSL buffer slot for the runtime-array sizes buffer.
    ///
    /// `None` means no runtime-array bindings are present in this entry
    /// point and naga does not emit a `_mslBufferSizes` struct.
    pub(crate) sizes_slot: Option<u64>,
    /// Bindings that contribute a `uint sizeN;` field to `_mslBufferSizes`,
    /// in declaration order. At dispatch time we look up the byte length of
    /// each bound buffer and pack them into a `Vec<u32>` that is bound at
    /// [`sizes_slot`](Self::sizes_slot) via `setBytes`.
    pub(crate) sizes_bindings: Arc<Vec<(u32, u32)>>,
    /// Threadgroup memory sizes (in bytes) to allocate, indexed by the
    /// implicit MSL threadgroup buffer index assigned by naga (declaration
    /// order). At dispatch we call `setThreadgroupMemoryLength:atIndex:`
    /// for each entry.
    pub(crate) threadgroup_sizes: Arc<Vec<u32>>,
    /// Workgroup size declared in the shader. Reserved for indirect dispatch.
    #[allow(dead_code)]
    pub(crate) workgroup_size: [u32; 3],
}

// SAFETY: ComputePipelineState wraps MTLComputePipelineState, which is thread-safe.
unsafe impl Send for MetalFunction {}
unsafe impl Sync for MetalFunction {}

// ── Encoder / Pass ─────────────────────────────────────────────────────

/// Metal command encoder. Owns the in-flight `MTLCommandBuffer`.
pub struct MetalEncoder {
    pub(crate) command_buffer: CommandBuffer,
}

unsafe impl Send for MetalEncoder {}
unsafe impl Sync for MetalEncoder {}

/// An active Metal compute pass. Holds the `MTLComputeCommandEncoder`
/// for the duration of the pass and ends encoding on drop.
pub struct MetalPass {
    pub(crate) encoder: ComputeCommandEncoder,
}

unsafe impl Send for MetalPass {}
unsafe impl Sync for MetalPass {}

impl Drop for MetalPass {
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

impl MetalPass {
    /// Inserts a buffer-scope memory barrier into this compute pass.
    ///
    /// metal-rs 0.32 doesn't wrap `memoryBarrierWithScope:`, so the
    /// selector is invoked directly via `msg_send!`.
    pub fn memory_barrier(&mut self) {
        const MTL_BARRIER_SCOPE_BUFFERS: u64 = 1;
        unsafe {
            let _: () = msg_send![self.encoder.as_ref(),
                memoryBarrierWithScope: MTL_BARRIER_SCOPE_BUFFERS];
        }
    }
}

// ── Dispatch ───────────────────────────────────────────────────────────

/// Collects buffer arguments and launches a Metal compute kernel.
pub struct MetalDispatch<'a> {
    pub(crate) encoder: &'a ComputeCommandEncoder,
    pub(crate) function: &'a MetalFunction,
    /// Collected (binding, buffer ref, byte_offset, byte_len) tuples.
    pub(crate) args: Vec<(ShaderBinding, &'a MtlBuffer, u64, u64)>,
    #[cfg(feature = "push_constants")]
    pub(crate) push_constants: Vec<u8>,
}

impl<'a> MetalDispatch<'a> {
    /// Adds a buffer argument at the given binding location.
    pub fn set_arg(
        &mut self,
        binding: ShaderBinding,
        buffer: &'a MtlBuffer,
        byte_offset: u64,
        byte_len: u64,
    ) {
        self.args.push((binding, buffer, byte_offset, byte_len));
    }

    /// Sets push constant data for this dispatch.
    #[cfg(feature = "push_constants")]
    pub fn set_push_constants(&mut self, data: &[u8]) {
        self.push_constants.clear();
        self.push_constants.extend_from_slice(data);
    }
}

// ── Timestamps ─────────────────────────────────────────────────────────

/// GPU timestamp manager backed by an `MTLCounterSampleBuffer` sampling at
/// stage boundaries (begin/end of each compute pass).
///
/// Each `begin_pass` allocates a `(start_index, end_index)` pair and wires
/// up the active compute pass descriptor's sample buffer attachment so
/// Metal records the GPU timestamp at the start and end of the encoder.
pub struct MetalTimestamps {
    sample_buffer: CounterSampleBuffer,
    /// `capacity * 2` total slots (one begin + one end per pass).
    capacity: u32,
    /// Number of `(begin, end)` pairs allocated so far this frame.
    next_index: u32,
    /// Pass labels in allocation order; aligned with sample-pair indices.
    labels: Vec<String>,
    /// Tick → nanosecond multiplier captured from the backend at creation.
    period_ns: f64,
    /// Completion flag for the fence committed by [`request_read`](Self::request_read),
    /// flipped by its command-buffer completion handler. Because the queue
    /// completes buffers in commit order, this being set means every earlier
    /// buffer — including the ones that sampled these timestamps — has finished
    /// writing the shared sample buffer.
    done: Option<Arc<AtomicBool>>,
}

// SAFETY: CounterSampleBuffer wraps an MTLCounterSampleBuffer, thread-safe.
unsafe impl Send for MetalTimestamps {}
unsafe impl Sync for MetalTimestamps {}

impl MetalTimestamps {
    /// Creates a new timestamp manager with room for `capacity` timed
    /// passes. Returns `None` if the device doesn't expose stage-boundary
    /// timestamp sampling, or if allocating the sample buffer fails.
    pub fn new(metal: &Metal, capacity: u32) -> Option<Self> {
        if capacity == 0 {
            return None;
        }
        let caps = metal.timing_caps.as_ref()?;
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&caps.counter_set);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count((capacity as u64) * 2);
        let sample_buffer = metal
            .device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .ok()?;
        Some(MetalTimestamps {
            sample_buffer,
            capacity,
            next_index: 0,
            labels: Vec::with_capacity(capacity as usize),
            period_ns: caps.period_ns,
            done: None,
        })
    }

    /// Resets the manager for a new frame (drops all collected pairs).
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.labels.clear();
        self.done = None;
    }

    /// Whether no non-blocking readback is in flight (safe to record a new frame).
    pub fn is_idle(&self) -> bool {
        self.done.is_none()
    }

    /// Initiates a non-blocking readback of the sampled timestamps.
    ///
    /// Call once after the frame's passes have been submitted. Commits an empty
    /// fence command buffer and returns immediately; poll for completion with
    /// [`try_take`](Self::try_take).
    pub fn request_read(&mut self, metal: &Metal) {
        self.done = Some(metal.commit_completion());
    }

    /// Non-blocking poll of a readback started by [`request_read`](Self::request_read).
    ///
    /// Returns `Some(results)` once the fence's completion handler has fired (so
    /// the sampled values are valid), or `None` while the GPU is still running.
    pub fn try_take(&mut self) -> Option<Vec<GpuTimestamp>> {
        let ready = self
            .done
            .as_ref()
            .is_some_and(|d| d.load(Ordering::Acquire));
        if !ready {
            return None;
        }
        self.done = None;
        Some(self.read().unwrap_or_default())
    }

    /// Reads back timestamp results after GPU synchronization.
    ///
    /// Must be called after the encoder containing the timed passes has
    /// been submitted *and* the device has been synchronized — otherwise
    /// the resolved values are unspecified.
    pub fn read(&self) -> Result<Vec<GpuTimestamp>, MetalBackendError> {
        if self.next_index == 0 {
            return Ok(Vec::new());
        }
        let count = self.next_index as u64 * 2;
        // metal-rs 0.32's `resolve_counter_range` wrapper has a bug where
        // it always passes length=0 to `getBytes:length:`, so we call
        // `resolveCounterRange:` ourselves and copy the bytes out manually.
        // Each `MTLCounterResultTimestamp` is a single u64 tick value.
        let mut raw = vec![0u64; count as usize];
        unsafe {
            let range = NSRange {
                location: 0,
                length: count,
            };
            let ns_data: *mut Object =
                msg_send![self.sample_buffer.as_ref(), resolveCounterRange: range];
            if !ns_data.is_null() {
                let total_bytes = count * std::mem::size_of::<u64>() as u64;
                let _: () = msg_send![ns_data,
                    getBytes: raw.as_mut_ptr() as *mut std::ffi::c_void
                    length: total_bytes];
            }
        }
        let mut entries = Vec::with_capacity(self.labels.len());
        for (i, label) in self.labels.iter().enumerate() {
            let begin = raw.get(i * 2).copied().unwrap_or(0);
            let end = raw.get(i * 2 + 1).copied().unwrap_or(0);
            let ticks = end.saturating_sub(begin) as f64;
            entries.push(GpuTimestamp {
                label: label.clone(),
                duration_ms: ticks * self.period_ns / 1_000_000.0,
            });
        }
        Ok(entries)
    }

    /// Allocates a `(begin, end)` index pair for a labeled pass. Returns
    /// `None` if the sample buffer is full.
    fn alloc_pair(&mut self, label: &str) -> Option<(NSUInteger, NSUInteger)> {
        if self.next_index >= self.capacity {
            return None;
        }
        let begin = (self.next_index * 2) as NSUInteger;
        let end = begin + 1;
        self.next_index += 1;
        self.labels.push(label.to_string());
        Some((begin, end))
    }
}

// ── Backend trait impl ─────────────────────────────────────────────────

impl Backend for Metal {
    const NAME: &'static str = "metal";
    const TARGET: CompileTarget = CompileTarget::Spirv;

    type Error = MetalBackendError;
    type Buffer<T: DeviceValue> = MetalBuffer<T>;
    type BufferSlice<'b, T: DeviceValue> = MetalBufferSlice<'b>;
    type Encoder = MetalEncoder;
    type Pass = MetalPass;
    type Timestamps = MetalTimestamps;
    type Module = MetalModule;
    type Function = MetalFunction;
    type Dispatch<'a> = MetalDispatch<'a>;

    fn as_metal(&self) -> Option<&Metal> {
        Some(self)
    }

    /*
     * Module / function loading.
     */
    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        // Module cache is keyed by content hash so the same SPIR-V isn't reparsed.
        let hash = fxhash(bytes);
        {
            let cache = self.module_cache.lock().unwrap();
            if let Some(module) = cache.get(&hash) {
                return Ok(module.clone());
            }
        }

        // Validate SPIR-V magic number.
        if bytes.len() < 4
            || u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0x07230203
        {
            return Err(MetalBackendError::SpirVParse(
                "Input does not start with SPIR-V magic number".into(),
            ));
        }

        // SPIR-V → naga IR.
        let module = naga::front::spv::parse_u8_slice(
            bytes,
            &naga::front::spv::Options {
                adjust_coordinate_space: false,
                strict_capabilities: false,
                block_ctx_dump_prefix: None,
            },
        )
        .map_err(|e| MetalBackendError::SpirVParse(format!("{e}")))?;

        // Validate so the MSL backend has the type info it needs.
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| MetalBackendError::NagaValidation(format!("{:?}", e.into_inner())))?;

        // Type layouts let us size threadgroup memory at dispatch time.
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .map_err(|e| MetalBackendError::NagaValidation(format!("layout: {e}")))?;

        let metal_module = MetalModule {
            naga: Arc::new(module),
            info: Arc::new(info),
            layouter: Arc::new(layouter),
        };

        self.module_cache
            .lock()
            .unwrap()
            .insert(hash, metal_module.clone());

        Ok(metal_module)
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
    ) -> Result<Self::Function, Self::Error> {
        // Without explicit layout info we infer bindings by scanning the naga module.
        let layouts = layouts_from_module(&module.naga);
        self.load_function_with_layouts(module, entry_point, push_constant_size, &layouts)
    }

    fn load_function_with_layouts(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
        layouts: &BindGroupLayoutInfo,
    ) -> Result<Self::Function, Self::Error> {
        // Collect all bindings, sort by (group, binding), assign MSL slots 0..N-1.
        let mut all_bindings: Vec<ShaderBinding> = layouts
            .groups
            .iter()
            .flat_map(|g| g.iter().copied())
            .collect();
        all_bindings.sort_by_key(|b| (b.space, b.index));

        let mut resources: BTreeMap<naga::ResourceBinding, naga::back::msl::BindTarget> =
            BTreeMap::new();
        let mut slot_map: Vec<(u32, u32, u64)> = Vec::with_capacity(all_bindings.len());
        for (slot, binding) in all_bindings.iter().enumerate() {
            let mutable = matches!(
                binding.descriptor_type,
                DescriptorType::Storage { read_only: false }
            );
            let target = naga::back::msl::BindTarget {
                buffer: Some(slot as u8),
                texture: None,
                sampler: None,
                external_texture: None,
                mutable,
            };
            resources.insert(
                naga::ResourceBinding {
                    group: binding.space,
                    binding: binding.index,
                },
                target,
            );
            slot_map.push((binding.space, binding.index, slot as u64));
        }

        let next_slot = all_bindings.len() as u8;
        #[cfg(feature = "push_constants")]
        let push_constant_slot: Option<u8> = if push_constant_size > 0 {
            Some(next_slot)
        } else {
            None
        };
        #[cfg(not(feature = "push_constants"))]
        let push_constant_slot: Option<u8> = None;
        let _ = push_constant_size;

        let next_slot = next_slot + push_constant_slot.is_some() as u8;

        // Collect the bindings that need a `uint sizeN;` entry in
        // `_mslBufferSizes`, in the same handle iteration order naga uses.
        // Each runtime-sized storage buffer contributes one entry.
        let mut sizes_bindings: Vec<(u32, u32)> = Vec::new();
        // Threadgroup-memory globals: collect their byte sizes in handle
        // iteration order. naga emits them as kernel parameters with no
        // `[[threadgroup(N)]]` attribute, so MSL assigns them indices
        // 0, 1, 2... in declaration order — matching this Vec's indices.
        let mut threadgroup_sizes: Vec<u32> = Vec::new();
        for (_, var) in module.naga.global_variables.iter() {
            if needs_array_length(var.ty, &module.naga.types)
                && let Some(b) = &var.binding
            {
                sizes_bindings.push((b.group, b.binding));
            }
            if matches!(var.space, naga::AddressSpace::WorkGroup) {
                let layout = module.layouter[var.ty];
                threadgroup_sizes.push(layout.size);
            }
        }
        let sizes_slot = if sizes_bindings.is_empty() {
            None
        } else {
            Some(next_slot)
        };

        let entry_point_resources = naga::back::msl::EntryPointResources {
            resources,
            immediates_buffer: push_constant_slot,
            sizes_buffer: sizes_slot,
        };

        let mut per_entry_point: BTreeMap<String, naga::back::msl::EntryPointResources> =
            BTreeMap::new();
        per_entry_point.insert(entry_point.to_string(), entry_point_resources);

        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            per_entry_point_map: per_entry_point,
            inline_samplers: vec![],
            spirv_cross_compatibility: false,
            fake_missing_bindings: false,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            zero_initialize_workgroup_memory: false,
            force_loop_bounding: false,
        };

        let pipeline_options = naga::back::msl::PipelineOptions {
            entry_point: Some((naga::ShaderStage::Compute, entry_point.to_string())),
            allow_and_force_point_size: false,
            vertex_pulling_transform: false,
            vertex_buffer_mappings: vec![],
        };

        let mut msl = String::new();
        let mut writer = naga::back::msl::Writer::new(&mut msl);
        let translation_info = writer
            .write(&module.naga, &module.info, &options, &pipeline_options)
            .map_err(|e| MetalBackendError::MslWrite(format!("{e}")))?;

        if std::env::var("KHAL_METAL_DUMP_MSL").is_ok() {
            eprintln!("──── MSL for `{}` ────\n{}\n────────────", entry_point, msl);
        }

        // Find the mangled MSL entry point name corresponding to our requested entry point.
        let mangled = translation_info
            .entry_point_names
            .iter()
            .zip(module.naga.entry_points.iter())
            .find_map(|(name_result, ep)| {
                if ep.name == entry_point {
                    name_result.as_ref().ok().cloned()
                } else {
                    None
                }
            })
            .ok_or_else(|| MetalBackendError::EntryPointNotFound(entry_point.into()))?;

        // Workgroup size lives in the entry point.
        let workgroup_size = module
            .naga
            .entry_points
            .iter()
            .find(|ep| ep.name == entry_point)
            .map(|ep| ep.workgroup_size)
            .ok_or_else(|| MetalBackendError::EntryPointNotFound(entry_point.into()))?;

        // Compile MSL.
        let compile_options = metal::CompileOptions::new();
        let library: Library = self
            .device
            .new_library_with_source(&msl, &compile_options)
            .map_err(MetalBackendError::LibraryCompile)?;

        let function = library
            .get_function(&mangled, None)
            .map_err(|e| MetalBackendError::EntryPointNotFound(format!("{entry_point}: {e}")))?;

        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        descriptor.set_label(entry_point);

        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor)
            .map_err(MetalBackendError::PipelineCreate)?;

        Ok(MetalFunction {
            pipeline,
            slot_map: Arc::new(slot_map),
            #[cfg(feature = "push_constants")]
            push_constant_slot: push_constant_slot.map(|s| s as u64),
            sizes_slot: sizes_slot.map(|s| s as u64),
            sizes_bindings: Arc::new(sizes_bindings),
            threadgroup_sizes: Arc::new(threadgroup_sizes),
            workgroup_size,
        })
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        let cmd_buf = self.queue.new_command_buffer().to_owned();
        MetalEncoder {
            command_buffer: cmd_buf,
        }
    }

    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a> {
        pass.encoder.set_compute_pipeline_state(&function.pipeline);
        MetalDispatch {
            encoder: &pass.encoder,
            function,
            args: Vec::new(),
            #[cfg(feature = "push_constants")]
            push_constants: Vec::new(),
        }
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        // Submit and wait on a fresh empty command buffer to flush the queue.
        let cb = self.queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        encoder.command_buffer.commit();
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
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let len = data.len();
        let byte_len = bytes.len();
        let inner = if byte_len == 0 {
            // MTLBuffer of length 0 isn't allowed; allocate a single byte placeholder.
            self.device.new_buffer(1, resource_options(usage))
        } else {
            self.device.new_buffer_with_data(
                bytes.as_ptr() as _,
                byte_len as NSUInteger,
                resource_options(usage),
            )
        };
        Ok(MetalBuffer {
            inner,
            len,
            byte_len,
            usage,
            _marker: PhantomData,
        })
    }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let elt_size = std::mem::size_of::<T>();
        let byte_len = (len * elt_size).max(1);
        let inner = self
            .device
            .new_buffer(byte_len as NSUInteger, resource_options(usage));
        Ok(MetalBuffer {
            inner,
            len,
            byte_len,
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
        let elt_size = std::mem::size_of::<T>();
        let byte_offset = (offset as usize) * elt_size;
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: contents() is valid for the buffer's lifetime; we copy non-overlapping bytes.
        unsafe {
            let dst = (buffer.inner.contents() as *mut u8).add(byte_offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        // Modified-range hint helps the driver on managed/storage modes.
        let range = metal::NSRange {
            location: byte_offset as NSUInteger,
            length: bytes.len() as NSUInteger,
        };
        buffer.inner.did_modify_range(range);
        Ok(())
    }

    async fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        // Make sure all prior submissions are visible.
        self.synchronize()?;
        if buffer.byte_len == 0 {
            return Ok(());
        }
        // SAFETY: contents() is valid for the buffer's lifetime; we copy non-overlapping bytes.
        let out_bytes = std::mem::size_of_val(out);
        let copy_len = buffer.byte_len.min(out_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.inner.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                copy_len,
            );
        }
        Ok(())
    }

    async fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        // For shared-storage MTLBuffers, host pointer access is direct after sync,
        // so this is identical to `read_buffer`. We blit through a staging buffer
        // when the source isn't host-coherent (private storage).
        if buffer.inner.storage_mode() == metal::MTLStorageMode::Private {
            // Allocate shared staging, blit, sync, copy out.
            let staging = self.uninit_buffer::<u8>(
                buffer.byte_len,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?;
            let encoder = self.begin_encoding();
            let blit = encoder.command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &buffer.inner,
                0,
                &staging.inner,
                0,
                buffer.byte_len as NSUInteger,
            );
            blit.end_encoding();
            self.submit(encoder)?;
            self.synchronize()?;

            let out_bytes = std::mem::size_of_val(out);
            let copy_len = staging.byte_len.min(out_bytes);
            // SAFETY: contents() valid for staging's lifetime; copy non-overlapping.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging.inner.contents() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    copy_len,
                );
            }
            return Ok(());
        }
        self.read_buffer(buffer, out).await
    }
}

// ── Encoder ────────────────────────────────────────────────────────────

impl Encoder<Metal> for MetalEncoder {
    fn begin_pass(&mut self, label: &str, timestamps: Option<&mut MetalTimestamps>) -> MetalPass {
        // If timestamps are requested and we still have room in the sample
        // buffer, configure the compute pass descriptor's
        // sample-buffer-attachment so Metal records GPU timestamps at the
        // start/end of this encoder.
        let encoder = match timestamps.and_then(|ts| {
            let (begin, end) = ts.alloc_pair(label)?;
            Some((ts, begin, end))
        }) {
            Some((ts, begin, end)) => {
                let descriptor = ComputePassDescriptor::new();
                descriptor.set_dispatch_type(MTLDispatchType::Serial);
                let attachment = descriptor
                    .sample_buffer_attachments()
                    .object_at(0)
                    .expect("compute pass sample buffer attachment 0");
                attachment.set_sample_buffer(&ts.sample_buffer);
                attachment.set_start_of_encoder_sample_index(begin);
                attachment.set_end_of_encoder_sample_index(end);
                self.command_buffer
                    .compute_command_encoder_with_descriptor(descriptor)
                    .to_owned()
            }
            None => self
                .command_buffer
                .compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial)
                .to_owned(),
        };
        if !label.is_empty() {
            encoder.set_label(label);
        }
        MetalPass { encoder }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &<Metal as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <Metal as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), MetalBackendError> {
        if copy_len == 0 {
            return Ok(());
        }
        let elt_size = std::mem::size_of::<T>();
        let blit = self.command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &source.inner,
            (source_offset * elt_size) as NSUInteger,
            &target.inner,
            (target_offset * elt_size) as NSUInteger,
            (copy_len * elt_size) as NSUInteger,
        );
        blit.end_encoding();
        Ok(())
    }

    fn memory_barrier(&mut self, pass: &mut MetalPass) {
        pass.memory_barrier();
    }
}

// ── Dispatch ───────────────────────────────────────────────────────────

impl<'a> Dispatch<'a, Metal> for MetalDispatch<'a> {
    #[cfg(feature = "push_constants")]
    fn set_push_constants(&mut self, data: &[u8]) {
        self.push_constants.clear();
        self.push_constants.extend_from_slice(data);
    }

    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, Metal>>,
        block_dim: [u32; 3],
    ) -> Result<(), MetalBackendError> {
        // Bind buffers per the (group, binding) → MSL slot map.
        for (binding, buffer, byte_offset, _byte_len) in &self.args {
            let slot = self
                .function
                .slot_map
                .iter()
                .find(|(g, b, _)| *g == binding.space && *b == binding.index)
                .map(|(_, _, s)| *s);
            if let Some(slot) = slot {
                self.encoder
                    .set_buffer(slot as NSUInteger, Some(buffer), *byte_offset);
            }
        }

        // Push constants: bind inline via setBytes at the reserved slot.
        #[cfg(feature = "push_constants")]
        if let Some(slot) = self.function.push_constant_slot {
            if !self.push_constants.is_empty() {
                self.encoder.set_bytes(
                    slot as NSUInteger,
                    self.push_constants.len() as NSUInteger,
                    self.push_constants.as_ptr() as *const _,
                );
            }
        }

        // Threadgroup memory: naga emits WorkGroup-space globals as kernel
        // parameters without explicit `[[threadgroup(N)]]` attributes, so
        // MSL assigns implicit indices in declaration order. The host must
        // size each slot via `setThreadgroupMemoryLength:atIndex:`.
        for (idx, &size) in self.function.threadgroup_sizes.iter().enumerate() {
            // Metal requires non-zero, 16-byte-aligned threadgroup sizes.
            let aligned = ((size as NSUInteger) + 15) & !15;
            let aligned = aligned.max(16);
            self.encoder
                .set_threadgroup_memory_length(idx as NSUInteger, aligned);
        }

        // Runtime-array sizes buffer: naga emits bounds checks against
        // `_mslBufferSizes.sizeN` for each storage buffer with a runtime
        // array. We pack the byte length of each such buffer (in the order
        // naga declared them) and bind via setBytes at `sizes_slot`.
        if let Some(slot) = self.function.sizes_slot {
            let mut sizes: smallvec::SmallVec<[u32; 8]> =
                smallvec::SmallVec::with_capacity(self.function.sizes_bindings.len());
            for (group, binding) in self.function.sizes_bindings.iter() {
                let entry = self
                    .args
                    .iter()
                    .find(|(b, _, _, _)| b.space == *group && b.index == *binding);
                let byte_len = entry
                    .map(|(_, _, _, byte_len)| *byte_len as u32)
                    .unwrap_or(0);
                sizes.push(byte_len);
            }
            if !sizes.is_empty() {
                self.encoder.set_bytes(
                    slot as NSUInteger,
                    (sizes.len() * std::mem::size_of::<u32>()) as NSUInteger,
                    sizes.as_ptr() as *const _,
                );
            }
        }

        // Resolve grid dimensions.
        let (grid_size, threads_per_threadgroup) = match grid.into() {
            DispatchGrid::Grid(g) => (
                MTLSize {
                    width: g[0] as NSUInteger,
                    height: g[1] as NSUInteger,
                    depth: g[2] as NSUInteger,
                },
                MTLSize {
                    width: block_dim[0] as NSUInteger,
                    height: block_dim[1] as NSUInteger,
                    depth: block_dim[2] as NSUInteger,
                },
            ),
            DispatchGrid::ThreadCount(t) => (
                MTLSize {
                    width: t[0].div_ceil(block_dim[0]) as NSUInteger,
                    height: t[1].div_ceil(block_dim[1]) as NSUInteger,
                    depth: t[2].div_ceil(block_dim[2]) as NSUInteger,
                },
                MTLSize {
                    width: block_dim[0] as NSUInteger,
                    height: block_dim[1] as NSUInteger,
                    depth: block_dim[2] as NSUInteger,
                },
            ),
            DispatchGrid::Indirect(buffer) => {
                self.encoder.dispatch_thread_groups_indirect(
                    &buffer.inner,
                    0,
                    MTLSize {
                        width: block_dim[0] as NSUInteger,
                        height: block_dim[1] as NSUInteger,
                        depth: block_dim[2] as NSUInteger,
                    },
                );
                return Ok(());
            }
        };

        if grid_size.width == 0 || grid_size.height == 0 || grid_size.depth == 0 {
            return Ok(());
        }

        self.encoder
            .dispatch_thread_groups(grid_size, threads_per_threadgroup);

        Ok(())
    }
}

// ── Buffer trait impl ──────────────────────────────────────────────────

impl<T: DeviceValue> crate::backend::Buffer<Metal, T> for MetalBuffer<T> {
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        self.len
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> MetalBufferSlice<'_> {
        let elt_size = std::mem::size_of::<T>() as u64;
        let total = self.byte_len as u64;
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n as u64 * elt_size,
            std::ops::Bound::Excluded(&n) => (n as u64 + 1) * elt_size,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => (n as u64 + 1) * elt_size,
            std::ops::Bound::Excluded(&n) => n as u64 * elt_size,
            std::ops::Bound::Unbounded => total,
        };
        MetalBufferSlice {
            buffer: &self.inner,
            byte_offset: start,
            byte_len: end - start,
        }
    }

    fn usage(&self) -> BufferUsages {
        self.usage
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Maps khal usage flags to Metal resource options.
///
/// `MAP_READ`/`MAP_WRITE` → shared storage (CPU/GPU coherent).
/// Otherwise → private storage (GPU-only, fastest).
fn resource_options(usage: BufferUsages) -> MTLResourceOptions {
    if usage.intersects(BufferUsages::MAP_READ | BufferUsages::MAP_WRITE) {
        MTLResourceOptions::StorageModeShared
    } else {
        // Private would require staging for any host upload; use Shared by
        // default so init_buffer / write_buffer remain straightforward and
        // match the semantics of wgpu's queue.write_buffer.
        MTLResourceOptions::StorageModeShared
    }
}

/// Returns true if `ty` (or its trailing struct member) is a runtime-sized
/// array. naga emits a `uint sizeN;` field in `_mslBufferSizes` for each
/// global variable for which this returns true.
fn needs_array_length(ty: naga::Handle<naga::Type>, types: &naga::UniqueArena<naga::Type>) -> bool {
    match types[ty].inner {
        naga::TypeInner::Struct { ref members, .. } => {
            if let Some(member) = members.last()
                && let naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                } = types[member.ty].inner
            {
                return true;
            }
            false
        }
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => true,
        _ => false,
    }
}

/// Best-effort recovery of bind group layout info from a naga module.
/// Used as a fallback when [`Backend::load_function`] is called without
/// explicit layout info.
fn layouts_from_module(module: &naga::Module) -> BindGroupLayoutInfo {
    let mut groups: Vec<Vec<ShaderBinding>> = Vec::new();
    for (_, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else {
            continue;
        };
        let descriptor_type = match var.space {
            naga::AddressSpace::Uniform => DescriptorType::Uniform,
            naga::AddressSpace::Storage { access } => DescriptorType::Storage {
                read_only: !access.contains(naga::StorageAccess::STORE),
            },
            _ => continue,
        };
        let group = binding.group as usize;
        if groups.len() <= group {
            groups.resize_with(group + 1, Vec::new);
        }
        groups[group].push(ShaderBinding {
            space: binding.group,
            index: binding.binding,
            descriptor_type,
        });
    }
    BindGroupLayoutInfo { groups }
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
