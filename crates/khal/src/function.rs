use crate::backend::{
    Backend, Dispatch, DispatchGrid, GpuBackend, GpuBackendError, GpuDispatch, GpuFunction,
    GpuPass, InnerGpuFunction,
};
use crate::shader::{ShaderArgs, ShaderArgsType};
use std::marker::PhantomData;

impl<Args: ShaderArgsType> GpuFunction<Args> {
    /// Maximum number of workgroups per dispatch dimension (WebGPU limit).
    pub const MAX_NUM_WORKGROUPS: u32 = 65535;

    /// Creates a typed GpuFunction from raw SPIR-V bytes with an explicit entry point.
    ///
    /// The `Args` type parameter's associated `For<'a>` type is used to determine
    /// the push constant size and bind group layouts.
    pub fn from_bytes(
        backend: &GpuBackend,
        bytes: &[u8],
        entry_point: &str,
    ) -> Result<Self, GpuBackendError> {
        Self::from_bytes_with_passthrough(backend, bytes, entry_point, false)
    }

    /// Creates a typed GpuFunction from raw SPIR-V bytes with an explicit entry point,
    /// optionally using SPIR-V passthrough loading.
    ///
    /// When `spirv_passthrough` is true, the SPIR-V bytecode is passed directly to the
    /// GPU driver without naga validation/transpilation. This is useful for shaders that
    /// use SPIR-V features not supported by naga (e.g. scalar block layout).
    /// On the WebGPU backend, this requires the `SPIRV_SHADER_PASSTHROUGH` wgpu feature
    /// (available when wgpu uses the Vulkan backend). On the Vulkan (ash) backend, SPIR-V
    /// is always loaded natively, so this flag has no effect.
    pub fn from_bytes_with_passthrough(
        backend: &GpuBackend,
        bytes: &[u8],
        entry_point: &str,
        spirv_passthrough: bool,
    ) -> Result<Self, GpuBackendError> {
        let layouts = Args::For::<'static>::bind_group_layouts();
        let module = if spirv_passthrough {
            backend.load_module_bytes_spirv_passthrough(bytes)?
        } else {
            backend.load_module_bytes(bytes)?
        };
        let inner = backend.load_function_with_layouts(
            &module,
            entry_point,
            Args::For::<'static>::PUSH_CONSTANT_SIZE,
            &layouts,
        )?;
        Ok(Self::from_inner(inner))
    }

    /// Creates a GpuFunction from a backend-specific inner function.
    fn from_inner(inner: InnerGpuFunction) -> Self {
        match inner {
            #[cfg(feature = "webgpu")]
            InnerGpuFunction::WebGpu(f) => GpuFunction::WebGpu(f, PhantomData),
            #[cfg(feature = "cuda")]
            InnerGpuFunction::Cuda(f) => GpuFunction::Cuda(f, PhantomData),
            InnerGpuFunction::Noop => GpuFunction::Noop(PhantomData),
        }
    }

    /// Returns the inner function without the Args type parameter.
    pub fn inner(&self) -> InnerGpuFunction {
        match self {
            #[cfg(feature = "webgpu")]
            GpuFunction::WebGpu(f, _) => InnerGpuFunction::WebGpu(f.clone()),
            #[cfg(feature = "cuda")]
            GpuFunction::Cuda(f, _) => InnerGpuFunction::Cuda(f.clone()),
            GpuFunction::Noop(_) => InnerGpuFunction::Noop,
        }
    }

    fn bind_args<'a, 'b: 'a>(
        dispatch: &mut GpuDispatch<'a>,
        args: &'b impl ShaderArgs<'b>,
    ) -> Result<(), GpuBackendError> {
        args.write_arg((0, 0).into(), dispatch)?;
        Ok(())
    }

    /// Launches the function, clamping the dispatch size so it doesn't exceed WebGPU's 65535
    /// workgroup count limit.
    ///
    /// Only use this if your shader is capable of handling the case where it should have exceeded
    /// 65535 * WORKGROUP_SIZE.
    ///
    /// Panics if the shader's workgroup size isn't `1` along the second and third axes.
    pub fn launch_capped<'b>(
        &self,
        pass: &mut GpuPass,
        args: &'b Args::For<'b>,
        num_threads: u32,
    ) -> Result<(), GpuBackendError> {
        let block_dim = Args::WORKGROUP_SIZE;
        assert_eq!(
            block_dim[1], 1,
            "launch_capped isn't applicable in this case"
        );
        assert_eq!(
            block_dim[2], 1,
            "launch_capped isn't applicable in this case"
        );

        let max_num_threads = Self::MAX_NUM_WORKGROUPS * block_dim[0];
        self.launch_grid(
            pass,
            args,
            DispatchGrid::ThreadCount([num_threads.min(max_num_threads), 1, 1]),
        )
    }

    /// Launches the function with the specified dispatch grid.
    ///
    /// The grid can be:
    /// - A `u32` or `[u32; 3]` for thread-count-based dispatch (automatically divided by workgroup size)
    /// - A `DispatchGrid::Grid([u32; 3])` for explicit workgroup grid dispatch
    /// - A `&GpuBuffer<[u32; 3]>` or `&Tensor<[u32; 3]>` for indirect dispatch
    pub fn launch_grid<'b>(
        &self,
        pass: &mut GpuPass,
        args: &'b Args::For<'b>,
        grid: impl Into<DispatchGrid<'b, GpuBackend>>,
    ) -> Result<(), GpuBackendError> {
        let block_dim = Args::WORKGROUP_SIZE;
        let grid = grid.into().resolve(block_dim);
        let inner = self.inner();
        let mut dispatch = pass.begin_dispatch(&inner);
        Self::bind_args(&mut dispatch, args)?;
        dispatch.launch(grid, block_dim)?;
        Ok(())
    }
}
