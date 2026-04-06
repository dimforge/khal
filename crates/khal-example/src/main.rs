use khal::backend::{Backend, Buffer, Encoder, GpuBackend, GpuBackendError, WebGpu};
use khal::re_exports::include_dir::{Dir, include_dir};
use khal::{BufferUsages, Shader};
use khal_example_shaders::AddAssign;

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[derive(Shader)]
pub struct GpuKernels {
    add_assign: AddAssign,
}

#[async_std::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    // Run the operation and display the result.
    let a = (0..10000).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..10000).map(|i| i as f32 * 10.0).collect::<Vec<_>>();
    let result = compute_sum(&backend, &a, &b).await.unwrap();
    println!("Computed sum: {result:?}");
}

async fn compute_sum(
    backend: &GpuBackend,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, GpuBackendError> {
    // Generate the GPU buffers.
    let mut a = backend.init_buffer(a, BufferUsages::STORAGE | BufferUsages::COPY_SRC)?;
    let b = backend.init_buffer(b, BufferUsages::STORAGE)?;

    // Dispatch the operation on the gpu.
    let kernels = GpuKernels::from_backend(backend)?;
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("add_assign", None);
    kernels.add_assign.call(&mut pass, a.len(), &mut a, &b)?;
    drop(pass);
    backend.submit(encoder)?;

    // Read the result (slower but convenient version).
    backend.slow_read_vec(&a).await
}
