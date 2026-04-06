use crate::backend::{
    Backend, BufferUsages, DescriptorType, DeviceValue, Dispatch, DispatchGrid, Encoder,
    GpuBackendError, GpuTimestamp, MaybeSendSync, ShaderBinding,
};
use crate::shader::{BindGroupLayoutInfo, ShaderArgsError};
use async_channel::RecvError;
use bytemuck::{AnyBitPattern, NoUninit};
use regex::Regex;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ops::RangeBounds;
use std::sync::Arc;
#[cfg(feature = "push_constants")]
use wgpu::PushConstantRange;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::CommandEncoderDescriptor;
use wgpu::{
    Adapter, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
    BufferAddress, BufferBindingType, BufferDescriptor, BufferSlice, BufferView, CommandEncoder,
    ComputePass, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    ExperimentalFeatures, Instance, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PollError, Queue, ShaderModule, ShaderRuntimeChecks, ShaderStages,
};

/// A WebGPU buffer slice that tracks its byte length for safe reinterpretation.
#[derive(Clone, Copy)]
pub struct WebGpuBufferSlice<'a> {
    pub(crate) inner: BufferSlice<'a>,
    pub(crate) byte_len: u64,
}

impl<'a> From<WebGpuBufferSlice<'a>> for wgpu::BindingResource<'a> {
    fn from(slice: WebGpuBufferSlice<'a>) -> Self {
        slice.inner.into()
    }
}

/// Wrapper for ShaderModule that includes binding information needed for push constants.
#[cfg(feature = "push_constants")]
#[derive(Clone)]
pub struct WebGpuModule {
    pub module: ShaderModule,
    /// Binding entries extracted from the SPIR-V for creating explicit bind group layouts.
    pub bindings: Vec<BindGroupLayoutEntry>,
}

/// A loaded WebGPU shader module (alias for `wgpu::ShaderModule` without push constants).
#[cfg(not(feature = "push_constants"))]
pub type WebGpuModule = ShaderModule;

/// A WebGPU compute function with its bind group layouts.
#[derive(Clone)]
pub struct WebGpuFunction {
    pub pipeline: ComputePipeline,
    /// Bind group layouts indexed by set number. Empty vec means use auto-generated layouts.
    pub bind_group_layouts: Arc<Vec<BindGroupLayout>>,
}

/// A WebGPU compute pass that carries its own device reference.
pub struct WebGpuPass {
    pub(crate) pass: ComputePass<'static>,
    pub(crate) device: Device,
}

impl WebGpuPass {
    /// Begins a compute dispatch within this pass, binding the given function.
    pub fn begin_dispatch<'a>(&'a mut self, function: &'a WebGpuFunction) -> WebGpuDispatch<'a> {
        WebGpuDispatch::new(&self.device, &mut self.pass, function)
    }
}

/// A WebGPU command encoder that carries its own device reference.
pub struct WebGpuEncoder {
    pub(crate) encoder: CommandEncoder,
    pub(crate) device: Device,
}

/// Helper struct to initialize a device and its queue.
#[derive(Clone)]
pub struct WebGpu {
    _instance: Instance, // TODO: do we have to keep this around?
    _adapter: Adapter,   // TODO: do we have to keep this around?
    device: Device,
    queue: Queue,
    hacks: Vec<(Regex, String)>,
    /// If this flag is set, every buffer created by this backend will have the
    /// `BufferUsages::COPY_SRC` flag. Useful for debugging.
    pub force_buffer_copy_src: bool,
    /// Whether the `SPIRV_SHADER_PASSTHROUGH` feature is enabled on this device.
    spirv_passthrough_enabled: bool,
    /// Whether `TIMESTAMP_QUERY` is supported on this device.
    timestamp_supported: bool,
}

impl WebGpu {
    /// Creates a WebGPU backend with default features and limits.
    pub async fn default() -> anyhow::Result<Self> {
        Self::new(wgpu::Features::default(), wgpu::Limits::default()).await
    }

    #[allow(unused_mut)] // features and limits need mut depending on enabled cargo features.
    /// Initializes a wgpu instance and create its queue.
    pub async fn new(
        mut features: wgpu::Features,
        mut limits: wgpu::Limits,
    ) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to initialize gpu adapter."))?;
        #[cfg(feature = "push_constants")]
        {
            features = features | wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SUBGROUP;
            // 128 bytes is the guaranteed minimum on Vulkan.
            limits.max_push_constant_size = 128;
        }
        #[cfg(feature = "subgroup_ops")]
        {
            features = features | wgpu::Features::SUBGROUP;
        }

        // Enable timestamp queries if the adapter supports it.
        let timestamp_supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamp_supported {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        // Enable SPIRV passthrough if the adapter supports it AND the backend is Vulkan.
        // SPIR-V passthrough only makes sense on Vulkan where raw SPIR-V can be consumed
        // directly by the driver. On other backends (Metal, DX12, etc.), the SPIR-V would
        // still need to be transpiled, so passthrough would not work correctly.
        let is_vulkan_backend = adapter.get_info().backend == wgpu::Backend::Vulkan;
        let spirv_passthrough_enabled = is_vulkan_backend
            && adapter
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS);
        if spirv_passthrough_enabled {
            features |= wgpu::Features::PASSTHROUGH_SHADERS;
        }

        // SAFETY: we opt into experimental features to enable SPIRV passthrough.
        // We only use this for the passthrough shader loading path, which is itself unsafe.
        let experimental_features = if spirv_passthrough_enabled {
            unsafe { ExperimentalFeatures::enabled() }
        } else {
            ExperimentalFeatures::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features,
            })
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            force_buffer_copy_src: false,
            hacks: vec![],
            spirv_passthrough_enabled,
            timestamp_supported,
        })
    }

    /// Adds a regex-based text replacement to apply to WGSL source before compilation.
    pub fn append_hack(&mut self, regex: Regex, replace_pattern: String) {
        self.hacks.push((regex, replace_pattern));
    }

    /// The `wgpu` device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The `wgpu` queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Whether GPU timestamp queries are supported on this device.
    pub fn timestamp_supported(&self) -> bool {
        self.timestamp_supported
    }

    /// Loads a SPIR-V module, extracting binding info for push constants support.
    #[cfg(feature = "push_constants")]
    fn load_module_spirv(&self, spirv_bytes: &[u8]) -> Result<WebGpuModule, WebGpuBackendError> {
        let source = wgpu::util::make_spirv(spirv_bytes);
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source,
            });
        // TODO: Extract bindings from SPIR-V using naga for proper push_constants support
        Ok(WebGpuModule {
            module: shader_module,
            bindings: Vec::new(),
        })
    }

    /// Loads a SPIR-V module.
    #[cfg(not(feature = "push_constants"))]
    fn load_module_spirv(&self, spirv_bytes: &[u8]) -> Result<WebGpuModule, WebGpuBackendError> {
        let source = wgpu::util::make_spirv(spirv_bytes);
        let shader_module = unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: None,
                    source,
                },
                ShaderRuntimeChecks::unchecked(),
            )
        };
        Ok(shader_module)
    }

    /// Loads a SPIR-V module using passthrough (bypassing naga validation/transpilation).
    ///
    /// Uses `create_shader_module_passthrough` to pass raw SPIR-V directly to the Vulkan driver.
    /// Falls back to normal SPIR-V loading if `EXPERIMENTAL_PASSTHROUGH_SHADERS` is not available.
    pub fn load_module_spirv_passthrough(
        &self,
        spirv_bytes: &[u8],
    ) -> Result<WebGpuModule, WebGpuBackendError> {
        if !self.spirv_passthrough_enabled {
            return self.load_module_spirv(spirv_bytes);
        }

        let spirv = wgpu::util::make_spirv_raw(spirv_bytes);

        // SAFETY: the caller has marked this shader as requiring passthrough loading,
        // meaning it uses SPIR-V features not supported by naga. The SPIR-V bytecode
        // is passed directly to the Vulkan driver without validation.
        let shader_module = unsafe {
            self.device
                .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    spirv: Some(spirv),
                    ..Default::default()
                })
        };

        #[cfg(feature = "push_constants")]
        return Ok(WebGpuModule {
            module: shader_module,
            bindings: Vec::new(),
        });

        #[cfg(not(feature = "push_constants"))]
        Ok(shader_module)
    }
}

/// Errors specific to the WebGPU backend.
#[derive(thiserror::Error, Debug)]
pub enum WebGpuBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    // #[error(transparent)]
    // Wgpu(#[from] wgpu::Error), // Doesn't implement Send+Sync
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error("Failed to read buffer from GPU: {0}")]
    BufferRead(RecvError),
    #[error(transparent)]
    DevicePoll(#[from] PollError),
    #[error(transparent)]
    Recv(#[from] RecvError),
    #[error("Failed to parse SPIR-V: {0}")]
    SpirVParse(String),
    #[error("Naga validation failed: {0}")]
    NagaValidation(String),
    #[error("Failed to write WGSL: {0}")]
    WgslWrite(String),
}

impl Backend for WebGpu {
    const NAME: &'static str = "webgpu";
    const TARGET: super::CompileTarget = super::CompileTarget::Wgsl;

    type Error = WebGpuBackendError;
    type Buffer<T: DeviceValue> = Buffer;
    type BufferSlice<'b, T: DeviceValue> = WebGpuBufferSlice<'b>;
    type Encoder = WebGpuEncoder;
    type Pass = WebGpuPass;
    type Timestamps = WebGpuTimestamps;
    type Module = WebGpuModule;
    type Function = WebGpuFunction;
    type Dispatch<'a> = WebGpuDispatch<'a>;

    fn as_webgpu(&self) -> Option<&WebGpu> {
        Some(self)
    }

    /*
     * Module/function loading.
     */
    fn load_module(&self, data: &str) -> Result<Self::Module, Self::Error> {
        // HACK: do we still need this hack? (probably not, that was needed when we relied on Slang).
        let mut data = data.replace("enable f16;", "").replace("f16", "f32");

        // Apply other user-defined hacks.
        for (reg, replace) in &self.hacks {
            data = reg.replace_all(&data, replace).to_string();
        }

        let shader_module = unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&data)),
                },
                ShaderRuntimeChecks::unchecked(),
            )
        };

        #[cfg(feature = "push_constants")]
        {
            // For WGSL, we don't extract bindings - push constants mode uses SPIR-V
            Ok(WebGpuModule {
                module: shader_module,
                bindings: Vec::new(),
            })
        }

        #[cfg(not(feature = "push_constants"))]
        Ok(shader_module)
    }

    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        // Check for SPIR-V magic number (0x07230203) at the start
        if bytes.len() >= 4 {
            let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if magic == 0x07230203 {
                return self.load_module_spirv(bytes);
            }
        }
        self.load_module(str::from_utf8(bytes).unwrap())
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
    ) -> Result<Self::Function, Self::Error> {
        // Use empty layout info - will use auto-generated layouts
        self.load_function_with_layouts(
            module,
            entry_point,
            push_constant_size,
            &BindGroupLayoutInfo::default(),
        )
    }

    fn load_function_with_layouts(
        &self,
        module: &Self::Module,
        entry_point: &str,
        push_constant_size: u32,
        layouts: &BindGroupLayoutInfo,
    ) -> Result<Self::Function, Self::Error> {
        // Create bind group layouts from the provided layout info
        let bind_group_layouts: Vec<BindGroupLayout> = layouts
            .groups
            .iter()
            .enumerate()
            .map(|(set_idx, bindings)| {
                let entries: Vec<BindGroupLayoutEntry> = bindings
                    .iter()
                    .map(|binding| BindGroupLayoutEntry {
                        binding: binding.index,
                        visibility: ShaderStages::COMPUTE,
                        ty: match binding.descriptor_type {
                            DescriptorType::Uniform => BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            DescriptorType::Storage { read_only } => BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        count: None,
                    })
                    .collect();

                self.device
                    .create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: Some(&format!("{}:set{}", entry_point, set_idx)),
                        entries: &entries,
                    })
            })
            .collect();

        // Create pipeline layout if we have explicit bind group layouts
        let (shader_module, pipeline_layout) = if !bind_group_layouts.is_empty() {
            let layout_refs: Vec<_> = bind_group_layouts.iter().map(Some).collect();

            #[cfg(feature = "push_constants")]
            let push_constant_ranges: Vec<PushConstantRange> = if push_constant_size > 0 {
                vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..push_constant_size,
                }]
            } else {
                vec![]
            };

            let layout = self
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some(entry_point),
                    bind_group_layouts: &layout_refs,
                    immediate_size: push_constant_size,
                });

            #[cfg(feature = "push_constants")]
            let sm = &module.module;
            #[cfg(not(feature = "push_constants"))]
            let sm = module;

            (sm, Some(layout))
        } else {
            // No explicit layouts - let wgpu auto-generate from shader
            #[cfg(feature = "push_constants")]
            let sm = &module.module;
            #[cfg(not(feature = "push_constants"))]
            let sm = module;

            let _ = push_constant_size;
            (sm, None)
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: pipeline_layout.as_ref(),
                module: shader_module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });

        Ok(WebGpuFunction {
            pipeline,
            bind_group_layouts: Arc::new(bind_group_layouts),
        })
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        WebGpuEncoder {
            encoder: self
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default()),
            device: self.device.clone(),
        }
    }

    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> WebGpuDispatch<'a> {
        pass.begin_dispatch(function)
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        let _ = self.queue.submit(Some(encoder.encoder.finish()));
        Ok(())
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + NoUninit>(
        &self,
        data: &[T],
        mut usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if self.force_buffer_copy_src && !usage.contains(BufferUsages::MAP_READ) {
            usage |= BufferUsages::COPY_SRC;
        }

        Ok(self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::try_cast_slice(data)?,
            usage: usage.into(),
        }))
    }

    // fn init_buffer_bytes<T: Copy>(&self, data: &[u8], usage: BufferUsages) -> Result<Self::Buffer<T>, Self::Error> {
    //     Ok(self.device.create_buffer_init(&BufferInitDescriptor {
    //         label: None,
    //         contents: data,
    //         usage,
    //     }))
    // }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        mut usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if self.force_buffer_copy_src && !usage.contains(BufferUsages::MAP_READ) {
            usage |= BufferUsages::COPY_SRC;
        }

        let bytes_len = std::mem::size_of::<T>() as u64 * len as u64;
        Ok(self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes_len,
            usage: usage.into(),
            mapped_at_creation: false,
        }))
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let elt_sz = std::mem::size_of::<T>() as u64;
        self.queue
            .write_buffer(buffer, offset * elt_sz, bytemuck::cast_slice(data));
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    async fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        let data = read_bytes(&self.device, buffer).await?;
        let out_bytes = core::mem::size_of_val(out);
        let copy_len = data.len().min(out_bytes);

        #[allow(dead_code)]
        if false {
            // NOTE: we keep this (but don’t actually run it) as a safeguad to
            //       ensure we don’t mistakenly remove the AnyBitPattern trait
            //       bound on T, which is required for soundness of the `copy_nonoverlapping`
            //       below.
            let _ = bytemuck::try_cast_slice::<_, T>(&data[..copy_len])?;
        }
        // SAFETY:
        // - `out` is valid, aligned, and large enough for `copy_len` bytes
        // - `T: AnyBitPattern` guarantees any byte pattern is a valid `T`
        // - Source and destination do not overlap (GPU mapped memory vs heap)
        // We don’t just use bytemuck::cast_slice because the `data` view might
        // not have the necessary alignment in all platforms.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr() as *mut u8, copy_len);
        }
        drop(data);
        buffer.unmap();
        Ok(())
    }

    async fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        // Create staging buffer.
        let bytes_len = buffer.size() as usize;
        let staging =
            self.uninit_buffer::<u8>(bytes_len, BufferUsages::MAP_READ | BufferUsages::COPY_DST)?;
        let mut encoder = self.begin_encoding();
        encoder
            .encoder
            .copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes_len as u64);
        self.submit(encoder)?;

        // Read the buffer.
        self.read_buffer(&staging, out).await
    }
}

impl Encoder<WebGpu> for WebGpuEncoder {
    fn begin_pass(&mut self, label: &str, timestamps: Option<&mut WebGpuTimestamps>) -> WebGpuPass {
        let mut desc = wgpu::ComputePassDescriptor {
            label: (!label.is_empty()).then_some(label),
            timestamp_writes: None,
        };

        if let Some(timestamps) = timestamps
            && let Some((begin_idx, end_idx)) = timestamps.alloc_timestamp_pair(label.to_string())
        {
            desc.timestamp_writes = Some(wgpu::ComputePassTimestampWrites {
                query_set: &timestamps.query_set,
                beginning_of_pass_write_index: Some(begin_idx),
                end_of_pass_write_index: Some(end_idx),
            });
        }

        WebGpuPass {
            pass: self.encoder.begin_compute_pass(&desc).forget_lifetime(),
            device: self.device.clone(),
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &<WebGpu as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <WebGpu as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), WebGpuBackendError> {
        wgpu::CommandEncoder::copy_buffer_to_buffer(
            &mut self.encoder,
            source,
            source_offset as BufferAddress * size_of::<T>() as BufferAddress,
            target,
            target_offset as BufferAddress * size_of::<T>() as BufferAddress,
            copy_len as BufferAddress * size_of::<T>() as BufferAddress,
        );
        Ok(())
    }
}

impl<'a> Dispatch<'a, WebGpu> for WebGpuDispatch<'a> {
    #[cfg(feature = "push_constants")]
    fn set_push_constants(&mut self, data: &[u8]) {
        self.push_constants.clear();
        self.push_constants.extend_from_slice(data);
    }

    // NOTE: the block_dim is configured in the shader…
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, WebGpu>>,
        _block_dim: [u32; 3],
    ) -> Result<(), WebGpuBackendError> {
        if !self.launchable {
            return Ok(());
        }

        self.pass.set_pipeline(&self.pipeline);

        // Set push constants if enabled and data is present.
        #[cfg(feature = "push_constants")]
        if !self.push_constants.is_empty() {
            self.pass.set_push_constants(0, &self.push_constants);
        }

        // Group bindings by descriptor set (space) and create bind groups.
        // Use pre-created layouts if available, otherwise query the pipeline.
        if !self.bind_group_layouts.is_empty() {
            // Use pre-created bind group layouts from ShaderArgs
            for (space, layout) in self.bind_group_layouts.iter().enumerate() {
                let entries: SmallVec<[_; 10]> = self
                    .args
                    .iter()
                    .filter(|(binding, _)| binding.space == space as u32)
                    .map(|(binding, input)| wgpu::BindGroupEntry {
                        binding: binding.index,
                        resource: input.inner.into(),
                    })
                    .collect();

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: &entries,
                });
                self.pass.set_bind_group(space as u32, &bind_group, &[]);
            }
        } else {
            // Fallback: query bind group layouts from the pipeline (auto-generated)
            let mut spaces: SmallVec<[u32; 4]> =
                self.args.iter().map(|(binding, _)| binding.space).collect();
            spaces.sort();
            spaces.dedup();

            for space in spaces {
                let entries: SmallVec<[_; 10]> = self
                    .args
                    .iter()
                    .filter(|(binding, _)| binding.space == space)
                    .map(|(binding, input)| wgpu::BindGroupEntry {
                        binding: binding.index,
                        resource: input.inner.into(),
                    })
                    .collect();

                // Try to get the bind group layout from the pipeline.
                let pipeline = &self.pipeline;
                let layout_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pipeline.get_bind_group_layout(space)
                }));

                if let Ok(layout) = layout_result {
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        entries: &entries,
                    });
                    self.pass.set_bind_group(space, &bind_group, &[]);
                }
            }
        }

        match grid.into() {
            DispatchGrid::Grid(grid_dim) => {
                // NOTE: we don't need to queue if the workgroup is empty.
                if grid_dim[0] * grid_dim[1] * grid_dim[2] > 0 {
                    self.pass
                        .dispatch_workgroups(grid_dim[0], grid_dim[1], grid_dim[2]);
                }
            }
            DispatchGrid::ThreadCount(threads) => {
                let grid_dim = [
                    threads[0].div_ceil(_block_dim[0]),
                    threads[1].div_ceil(_block_dim[1]),
                    threads[2].div_ceil(_block_dim[2]),
                ];
                if grid_dim[0] * grid_dim[1] * grid_dim[2] > 0 {
                    self.pass
                        .dispatch_workgroups(grid_dim[0], grid_dim[1], grid_dim[2]);
                }
            }
            DispatchGrid::Indirect(grid_indirect) => {
                self.pass.dispatch_workgroups_indirect(grid_indirect, 0);
            }
        }

        Ok(())
    }
}

/// Collects bind group entries and launches a WebGPU compute dispatch.
pub struct WebGpuDispatch<'a> {
    // NOTE: keep up to 10 bindings on the stack. This number was chosen to match
    //       the current (06/2025) max storage bindings on the browser.
    device: Device,
    pass: &'a mut ComputePass<'static>,
    pipeline: ComputePipeline,
    /// Pre-created bind group layouts from ShaderArgs. Empty means use auto-generated layouts.
    bind_group_layouts: Arc<Vec<BindGroupLayout>>,
    pub(crate) args: SmallVec<[(ShaderBinding, WebGpuBufferSlice<'a>); 10]>,
    launchable: bool,
    #[cfg(feature = "push_constants")]
    push_constants: SmallVec<[u8; 128]>,
}

impl<'a> WebGpuDispatch<'a> {
    fn new(
        device: &Device,
        pass: &'a mut ComputePass<'static>,
        function: &WebGpuFunction,
    ) -> WebGpuDispatch<'a> {
        WebGpuDispatch {
            device: device.clone(),
            pass,
            pipeline: function.pipeline.clone(),
            bind_group_layouts: function.bind_group_layouts.clone(),
            args: SmallVec::default(),
            launchable: true,
            #[cfg(feature = "push_constants")]
            push_constants: SmallVec::default(),
        }
    }
}

/// Extension trait for creating labeled compute passes from a command encoder.
#[allow(dead_code)]
pub trait CommandEncoderExt {
    /// Begins a labeled compute pass.
    fn compute_pass<'encoder>(&'encoder mut self, label: &str) -> ComputePass<'encoder>;
}

impl CommandEncoderExt for CommandEncoder {
    fn compute_pass<'encoder>(&'encoder mut self, label: &str) -> ComputePass<'encoder> {
        let desc = ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        };
        self.begin_compute_pass(&desc)
    }
}

impl CommandEncoderExt for WebGpuEncoder {
    fn compute_pass<'encoder>(&'encoder mut self, label: &str) -> ComputePass<'encoder> {
        self.encoder.compute_pass(label)
    }
}

async fn read_bytes(device: &Device, buffer: &Buffer) -> Result<BufferView, WebGpuBackendError> {
    let buffer_slice = buffer.slice(..);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let (sender, receiver) = async_channel::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send_blocking(v).unwrap()
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .await
            .map_err(WebGpuBackendError::BufferRead)?
            .unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    {
        let (sender, receiver) = async_channel::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.force_send(v).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv().await?.unwrap();
    }

    let data = buffer_slice.get_mapped_range();
    Ok(data)
}

impl<T: DeviceValue> crate::backend::Buffer<WebGpu, T> for Buffer {
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        self.size() as usize / std::mem::size_of::<T>()
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> <WebGpu as Backend>::BufferSlice<'_, T> {
        let elem_size = std::mem::size_of::<T>() as u64;
        let start_bytes = match range.start_bound() {
            std::ops::Bound::Included(&val) => val as u64 * elem_size,
            std::ops::Bound::Unbounded => 0,
            _ => unreachable!(),
        };
        let end_bytes = match range.end_bound() {
            std::ops::Bound::Excluded(&val) => val as u64 * elem_size,
            std::ops::Bound::Included(&val) => (val as u64 + 1) * elem_size,
            std::ops::Bound::Unbounded => self.size(),
        };
        WebGpuBufferSlice {
            inner: self.slice(start_bytes..end_bytes),
            byte_len: end_bytes - start_bytes,
        }
    }

    fn usage(&self) -> BufferUsages {
        self.usage().into()
    }
}

/// WebGPU timestamp query manager for profiling compute passes.
pub struct WebGpuTimestamps {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    capacity: u32,
    next_index: u32,
    labels: Vec<String>,
    timestamp_period: f32,
}

impl WebGpuTimestamps {
    /// Creates a new timestamp query manager with room for `capacity` timed passes.
    /// Returns `None` if the device does not support timestamp queries.
    pub fn new(backend: &WebGpu, capacity: u32) -> Option<Self> {
        if !backend.timestamp_supported {
            return None;
        }
        let query_count = capacity * 2; // begin + end per pass
        let query_set = backend.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64;
        let resolve_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timestamps_resolve"),
            size: resolve_buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timestamps_staging"),
            size: resolve_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let timestamp_period = backend.queue.get_timestamp_period();
        Some(WebGpuTimestamps {
            query_set,
            resolve_buffer,
            staging_buffer,
            capacity,
            next_index: 0,
            labels: Vec::with_capacity(capacity as usize),
            timestamp_period,
        })
    }

    /// Resets the query manager for a new frame.
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.labels.clear();
    }

    /// Resolves timestamp queries to a buffer and copies to staging for readback.
    pub fn resolve(&self, encoder: &mut WebGpuEncoder) {
        if self.next_index == 0 {
            // Nothing to resolve.
            return;
        }

        let query_count = self.next_index * 2;
        encoder
            .encoder
            .resolve_query_set(&self.query_set, 0..query_count, &self.resolve_buffer, 0);
        let copy_size = query_count as u64 * std::mem::size_of::<u64>() as u64;
        encoder.encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            copy_size,
        );
    }

    /// Reads back timestamp results after GPU synchronization.
    pub async fn read(&self, backend: &WebGpu) -> Result<Vec<GpuTimestamp>, GpuBackendError> {
        if self.next_index == 0 {
            return Ok(Vec::new());
        }
        let query_count = self.next_index * 2;
        let mut raw_timestamps = vec![0u64; query_count as usize];
        // Map and read the staging buffer.
        {
            use crate::backend::webgpu::WebGpuBackendError;
            let buffer_slice = self.staging_buffer.slice(..);
            let (sender, receiver) = async_channel::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = sender.force_send(v).unwrap();
            });
            backend
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(WebGpuBackendError::from)?;
            receiver
                .recv()
                .await
                .map_err(WebGpuBackendError::from)?
                .unwrap();
            let data = buffer_slice.get_mapped_range();
            let bytes = &*data;
            // SAFETY: u64 is AnyBitPattern and we ensured the buffer is large enough.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    raw_timestamps.as_mut_ptr() as *mut u8,
                    bytes
                        .len()
                        .min(raw_timestamps.len() * std::mem::size_of::<u64>()),
                );
            }
            drop(data);
            self.staging_buffer.unmap();
        }

        let period_ms = self.timestamp_period as f64 / 1_000_000.0;
        let results = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let begin = raw_timestamps[i * 2];
                let end = raw_timestamps[i * 2 + 1];
                let duration_ms = (end.wrapping_sub(begin)) as f64 * period_ms;
                GpuTimestamp {
                    label: label.clone(),
                    duration_ms,
                }
            })
            .collect();
        Ok(results)
    }

    /// Allocates a begin/end timestamp pair for a labeled pass.
    /// Returns `None` if there is not enough room for the extra pair.
    pub fn alloc_timestamp_pair(&mut self, label: String) -> Option<(u32, u32)> {
        if self.next_index < self.capacity {
            let begin_idx = self.next_index * 2;
            let end_idx = begin_idx + 1;
            self.next_index += 1;
            self.labels.push(label);
            Some((begin_idx, end_idx))
        } else {
            None
        }
    }
}
