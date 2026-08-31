use khal::backend::{Backend, DispatchGrid, Encoder, GpuBackend};
use khal::re_exports::include_dir::{Dir, include_dir};
use khal::{BufferUsages, Shader};

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");
use khal_example_shaders::{AddAssign, BatchProbe, GridProbe};

#[derive(Shader)]
pub struct GpuKernels {
    #[allow(dead_code)]
    add_assign: AddAssign,
    grid_probe: GridProbe,
    batch_probe: BatchProbe,
}

#[async_std::main]
async fn main() {
    let backend = GpuBackend::auto(Default::default(), Default::default())
        .await
        .unwrap();
    let kernels = GpuKernels::from_backend(&backend).unwrap();
    let nb = 4u32;
    let cap = 64u32;

    // grid_probe: bare thread-count dispatch [cap, nb, 1].
    let src: Vec<u32> = (0..nb).map(|b| 100 + b).collect();
    let src_buf = backend.init_buffer(&src, BufferUsages::STORAGE).unwrap();
    let mut out = backend
        .init_buffer(
            &vec![0u32; (cap * nb) as usize],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
    let mut enc = backend.begin_encoding();
    let mut pass = enc.begin_pass("probe", None);
    kernels
        .grid_probe
        .call(&mut pass, [cap, nb, 1], &src_buf, &mut out)
        .unwrap();
    drop(pass);
    backend.submit(enc).unwrap();
    let got: Vec<u32> = backend.slow_read_vec(&out).await.unwrap();
    let mut bad = 0;
    for y in 0..nb {
        for x in 0..cap {
            let want = (y << 20) | ((100 + y) << 10) | x;
            if got[(y * cap + x) as usize] != want {
                if bad < 5 {
                    println!(
                        "grid_probe MISMATCH y={y} x={x} got={:#x} want={want:#x}",
                        got[(y * cap + x) as usize]
                    );
                }
                bad += 1;
            }
        }
    }
    println!("grid_probe (threads form): {} mismatches", bad);

    // batch_probe: raw workgroup Grid dispatch [1, nb, 1] like nexus fixed grids.
    let lens: Vec<u32> = (0..nb).map(|b| 10 + b).collect();
    let lens_buf = backend.init_buffer(&lens, BufferUsages::STORAGE).unwrap();
    let src2: Vec<u32> = (0..cap * nb).collect();
    let src2_buf = backend.init_buffer(&src2, BufferUsages::STORAGE).unwrap();
    let mut out2 = backend
        .init_buffer(
            &vec![u32::MAX; (cap * nb) as usize],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
    let mut enc = backend.begin_encoding();
    let mut pass = enc.begin_pass("probe2", None);
    kernels
        .batch_probe
        .call(
            &mut pass,
            DispatchGrid::Grid([1, nb, 1]),
            &lens_buf,
            &src2_buf,
            &mut out2,
        )
        .unwrap();
    drop(pass);
    backend.submit(enc).unwrap();
    let got2: Vec<u32> = backend.slow_read_vec(&out2).await.unwrap();
    let mut bad2 = 0;
    for b in 0..nb {
        for i in 0..lens[b as usize] {
            let idx = (b * cap + i) as usize;
            let want = (b << 20) | src2[idx];
            if got2[idx] != want {
                if bad2 < 5 {
                    println!(
                        "batch_probe MISMATCH b={b} i={i} got={:#x} want={want:#x}",
                        got2[idx]
                    );
                }
                bad2 += 1;
            }
        }
    }
    println!("batch_probe (Grid form): {} mismatches", bad2);
}
