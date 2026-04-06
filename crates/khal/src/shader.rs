use crate::backend::{GpuBackend, GpuBackendError, GpuDispatch, ShaderBinding};

/// A collection of GPU compute functions that can be instantiated from a backend.
///
/// Typically derived with `#[derive(Shader)]`, which generates the `from_backend`
/// implementation that loads SPIR-V/PTX modules and creates all compute pipelines.
pub trait Shader: Sized + 'static {
    /// Instantiates `Self` and all its compute functions from a backend.
    fn from_backend(b: &GpuBackend) -> Result<Self, GpuBackendError>;
}

/// Trait for types that represent a "family" of ShaderArgs with different lifetimes.
///
/// This trait enables type-safe GPU function dispatch by connecting a marker type
/// to the concrete ShaderArgs type for any given lifetime. The marker type can be
/// used as a type parameter without carrying a lifetime.
///
/// This is typically derived automatically by `#[derive(ShaderArgs)]`.
///
/// # Example
///
/// ```ignore
/// // The ShaderArgs derive generates:
/// pub enum WarmstartArgsType {}
///
/// impl ShaderArgsType for WarmstartArgsType {
///     type For<'a> = WarmstartArgs<'a>;
/// }
/// ```
pub trait ShaderArgsType {
    /// The concrete ShaderArgs type for a given lifetime.
    type For<'a>: ShaderArgs<'a>;

    /// The workgroup size for this compute shader kernel.
    ///
    /// Defaults to `[1, 1, 1]`. When using `#[shader_args]`, this is automatically
    /// extracted from the `#[spirv(compute(threads(...)))]` attribute.
    const WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];
}

/// Implementation for the unit type, used as the default for untyped [`crate::backend::GpuFunction`].
impl ShaderArgsType for () {
    type For<'a> = ();
}

/// Errors that can occur when writing shader arguments to a dispatch.
#[derive(thiserror::Error, Debug)]
pub enum ShaderArgsError {
    #[error("argument not found: {0}")]
    ArgNotFound(String),
}

/// Information about bind group layouts derived from ShaderArgs.
/// Maps descriptor set index to binding entries.
#[derive(Clone, Debug, Default)]
pub struct BindGroupLayoutInfo {
    /// Bind group layouts indexed by set number. Each entry is a list of (binding_index, descriptor_type).
    pub groups: Vec<Vec<ShaderBinding>>,
}

/// Trait for types that can write shader arguments to a dispatch.
///
/// This trait is typically derived using `#[derive(ShaderArgs)]` with `#[bind(...)]`
/// and `#[push_constant]` field attributes:
///
/// ```ignore
/// #[derive(ShaderArgs)]
/// struct MyArgs<'a> {
///     #[bind(index = 0)]
///     input: &'a GpuBuffer<f32>,
///     #[bind(index = 1)]
///     output: &'a GpuBuffer<f32>,
///     #[push_constant]
///     shape_in: GpuTensorLayout,
///     #[push_constant]
///     shape_out: GpuTensorLayout,
/// }
/// ```
///
/// Fields marked with `#[push_constant]` are passed to the shader via push constants
/// (when the `push_constants` feature is enabled). The order of push constant fields
/// determines their layout in memory.
pub trait ShaderArgs<'b> {
    /// Size of push constants in bytes. 0 means no push constants.
    ///
    /// This is computed at compile time from fields marked with `#[push_constant]`.
    const PUSH_CONSTANT_SIZE: u32 = 0;

    /// Returns the bind group layout information for this shader args type.
    ///
    /// This describes all bindings grouped by descriptor set, allowing the backend
    /// to create explicit bind group layouts at pipeline creation time.
    fn bind_group_layouts() -> BindGroupLayoutInfo {
        BindGroupLayoutInfo::default()
    }

    /// Writes shader argument(s) to the dispatch.
    ///
    /// For leaf types (like buffers), writes itself at the given binding.
    /// For derived structs, writes all fields using their `#[bind(...)]` attributes
    /// (the `binding` parameter is ignored in this case).
    ///
    /// When push constants are enabled, also writes push constant data.
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a;
}

impl<'b> ShaderArgs<'b> for () {
    fn write_arg<'a>(
        &'b self,
        _binding: ShaderBinding,
        _dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        Ok(())
    }
}

impl<'b, T: ShaderArgs<'b>> ShaderArgs<'b> for Option<T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match self {
            Some(arg) => arg.write_arg(binding, dispatch),
            None => Ok(()),
        }
    }
}

impl<'b, T: ShaderArgs<'b>> ShaderArgs<'b> for &'b T {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        (*self).write_arg(binding, dispatch)
    }
}
