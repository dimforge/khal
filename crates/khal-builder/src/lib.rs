//! Build-time utilities for compiling shader crates to SPIR-V and PTX.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Configures and runs the SPIR-V and PTX shader compilation pipeline.
///
/// Used in `build.rs` scripts to compile a shader crate before the host crate.
pub struct KhalBuilder {
    shader_crate: PathBuf,
    // Useful for unusual crates layout where the src directory isn’t in `shader_crate/src`.
    shader_src: Option<PathBuf>,
    // Features to enable when building the library.
    features: Vec<String>,
    // The `RUST_MIN_STACK` given to the shader builders.
    rust_min_stack: u32,
    /// If the `cuda` feature is enabled and this is `true`, then cuda PTX kernels will be built
    /// with cargo-cuda (or cargo-oxide when the `cuda-oxide` feature is enabled).
    /// Default: `true`
    #[allow(dead_code)]
    build_cuda: bool,
    /// If this is `true`, then SpirV kernels will be built with cargo-gpu.
    /// Default: `true`
    build_spirv: bool,
}

impl KhalBuilder {
    /// Creates a new builder for the given shader crate directory.
    /// If `enable_builtin_features` is true, platform-specific features are auto-detected.
    pub fn new(shader_crate: impl AsRef<Path>, enable_builtin_features: bool) -> Self {
        let mut builder = Self {
            shader_crate: shader_crate.as_ref().to_owned(),
            shader_src: None,
            features: Vec::new(),
            build_cuda: true,
            build_spirv: true,
            rust_min_stack: 1024 * 1024 * 32,
        };
        if enable_builtin_features {
            builder = builder.append_builtin_features();
        }
        builder
    }

    /// Creates a new builder by locating the shader crate via cargo's `links`
    /// metadata mechanism.
    ///
    /// `links_name` must match the `links` value declared in the shader
    /// crate's `Cargo.toml`. The shader crate's `build.rs` must emit
    /// `cargo::metadata=manifest_dir=$CARGO_MANIFEST_DIR`, and the host crate
    /// must depend on the shader crate as a `[build-dependencies]` entry.
    /// Cargo then exposes `DEP_<LINKS>_MANIFEST_DIR` to this build script,
    /// which works identically for in-workspace path dependencies and for
    /// versions fetched from a registry.
    pub fn from_dependency(links_name: &str, enable_builtin_features: bool) -> Self {
        let env_key = format!(
            "DEP_{}_MANIFEST_DIR",
            links_name.to_ascii_uppercase().replace('-', "_")
        );
        let manifest_dir = std::env::var(&env_key).unwrap_or_else(|_| {
            panic!(
                "environment variable `{env_key}` is not set; ensure `{links_name}` is declared \
                 as a `[build-dependencies]` entry of the host crate and that its `build.rs` emits \
                 `cargo::metadata=manifest_dir=$CARGO_MANIFEST_DIR`"
            )
        });
        Self::new(manifest_dir, enable_builtin_features)
    }

    /// Sets the `RUST_MIN_STACK` environment variable for the shader compilation processes.
    pub fn rust_min_stack(mut self, stack: u32) -> Self {
        self.rust_min_stack = stack;
        self
    }

    /// Overrides the shader source directory (defaults to `<shader_crate>/src`).
    pub fn shader_src(mut self, src: impl AsRef<Path>) -> Self {
        self.shader_src = Some(src.as_ref().to_owned());
        self
    }

    /// Adds a cargo feature to enable when building the shader crate.
    pub fn feature(mut self, feature: impl ToString) -> Self {
        let feature = feature.to_string();
        if !self.features.contains(&feature) {
            self.features.push(feature);
        }
        self
    }

    /// Compiles the shader crate and writes output files to `output_dir`.
    pub fn build(self, output_dir: impl AsRef<Path>) {
        let output_dir = output_dir.as_ref();

        self.setup_change_detection();

        // Consumers embed this directory at compile time (e.g. vortx's
        // `include_dir!("$OUT_DIR/shaders-spirv")`), so it must exist even when the
        // SPIR-V build is skipped — otherwise the host crate fails to compile.
        std::fs::create_dir_all(output_dir)
            .unwrap_or_else(|e| panic!("failed to create shader output dir {output_dir:?}: {e}"));

        // `KHAL_SKIP_SPIRV=1` skips the cargo-gpu SPIR-V compile entirely. This lets
        // a CUDA-only build (shaders come from cuda-oxide PTX cubins) avoid the
        // cargo-gpu toolchain as a build prerequisite. The WebGPU backend then has
        // no embedded shaders and fails hard at runtime — intended on CUDA boxes.
        let skip_spirv = std::env::var_os("KHAL_SKIP_SPIRV").is_some();
        if self.build_spirv && !skip_spirv {
            self.build_spirv(output_dir);
        }

        #[cfg(feature = "cuda")]
        if self.build_cuda {
            self.build_ptx(output_dir);
        }
    }

    fn append_builtin_features(mut self) -> Self {
        if cfg!(feature = "unsafe_remove_boundchecks") {
            self = self.feature("unsafe-remove-boundchecks");
        }

        self
    }

    fn setup_change_detection(&self) {
        println!(
            "cargo:rerun-if-changed={}",
            self.shader_crate.to_string_lossy()
        );
        let shader_src = self
            .shader_src
            .clone()
            .unwrap_or_else(|| self.shader_crate.join("src"));
        for entry in walkdir::WalkDir::new(shader_src)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }

        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PUSH_CONSTANTS"); // TODO: currently unused
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_OXIDE");
        println!("cargo:rerun-if-env-changed=KHAL_SKIP_SPIRV");
    }

    fn build_spirv(&self, output_dir: impl AsRef<Path>) {
        let output_dir = output_dir.as_ref();
        let mut args = vec![
            "gpu",
            "build",
            "--shader-crate",
            self.shader_crate
                .to_str()
                .expect("Invalid shader crate path"),
            "--output-dir",
            output_dir.to_str().expect("Invalid output directory path"),
            "--multimodule",
        ];

        let features_str = self.features.join(",");
        if !features_str.is_empty() {
            args.push("--features");
            args.push(&features_str);
        }

        let status = Command::new("cargo")
            .args(args)
            .env("RUST_MIN_STACK", self.rust_min_stack.to_string())
            .status()
            .expect("failed to run cargo gpu");

        if !status.success() {
            panic!("cargo gpu build failed");
        }
    }

    /// Compiles the shader crate to PTX for the CUDA backend.
    ///
    /// `CUDA_OXIDE_SHADERS_PTX_<SHADER_CRATE_NAME>` (upper-snake, e.g.
    /// `CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS`) points at a prebuilt PTX/cubin;
    /// when set it is embedded directly as `shaders.ptx` and no PTX compiler
    /// (`cargo cuda` / `cargo oxide`) is required.
    ///
    /// Otherwise the PTX is produced by `cargo oxide` (cuda-oxide) when the
    /// `cuda-oxide` feature is enabled, and by `cargo cuda` (rust-cuda) when
    /// it is not.
    #[cfg(feature = "cuda")]
    fn build_ptx(&self, output_dir: impl AsRef<Path>) {
        let output_dir = output_dir.as_ref();

        let crate_name = self
            .shader_crate
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_uppercase()
            .replace('-', "_");
        let env_key = format!("CUDA_OXIDE_SHADERS_PTX_{crate_name}");
        println!("cargo:rerun-if-env-changed={env_key}");
        if let Some(prebuilt) = std::env::var_os(&env_key) {
            // Re-embed when the prebuilt file itself changes, not only its path.
            println!("cargo:rerun-if-changed={}", Path::new(&prebuilt).display());
            std::fs::create_dir_all(output_dir)
                .expect("failed to create shader output dir for the prebuilt cubin");
            let dst = output_dir.join("shaders.ptx");
            std::fs::copy(&prebuilt, &dst).unwrap_or_else(|e| {
                panic!("failed to copy prebuilt PTX {prebuilt:?} ({env_key}) to {dst:?}: {e}")
            });
            return;
        }

        #[cfg(feature = "cuda-oxide")]
        {
            self.build_ptx_cuda_oxide(output_dir);
        }
        #[cfg(not(feature = "cuda-oxide"))]
        {
            self.build_ptx_cargo_cuda(output_dir);
        }

        assemble_cubin(&output_dir.join("shaders.ptx"));
    }

    /// PTX via rust-cuda's `cargo cuda build` (rustc_codegen_nvvm).
    #[cfg(all(feature = "cuda", not(feature = "cuda-oxide")))]
    fn build_ptx_cargo_cuda(&self, output_dir: &Path) {
        let features_str = self.features.join(",");

        let mut args = vec![
            "cuda",
            "build",
            "--shader-crate",
            self.shader_crate
                .to_str()
                .expect("Invalid shader crate path"),
            "--output-dir",
            output_dir.to_str().expect("Invalid output directory path"),
        ];

        if !features_str.is_empty() {
            args.push("--features");
            args.push(&features_str);
        }

        let status = Command::new("cargo")
            .args(args)
            .env("RUST_MIN_STACK", self.rust_min_stack.to_string())
            .status()
            .expect("failed to run cargo cuda");

        if !status.success() {
            panic!("cargo cuda build failed");
        }
    }

    /// PTX via cuda-oxide's `cargo oxide build` (unified host-target interception).
    ///
    /// The shader crate is compiled as an ordinary host-target crate with its
    /// `cuda-oxide` feature enabled; the cuda-oxide codegen backend intercepts
    /// the `#[spirv_bindgen]`-generated kernel entries and writes
    /// `<crate_name>.ptx` into `CUDA_OXIDE_PTX_DIR`, which we then copy to
    /// `<output_dir>/shaders.ptx` (the file `from_dir_ptx` loads).
    ///
    /// Environment knobs (all optional):
    /// - `KHAL_CUDA_ARCH=sm_XX` / `CUDA_OXIDE_TARGET=sm_XX`: pin the PTX
    ///   target. Otherwise the local GPU's compute capability is detected with
    ///   `nvidia-smi`; if that fails, cuda-oxide's own default target is used.
    /// - `KHAL_CUDA_OXIDE_TARGET_DIR`: cargo target dir for the nested build.
    ///   Defaults to `<workspace target dir>/khal-cuda-oxide`. It must differ
    ///   from the outer build's target dir, which the outer cargo holds locked
    ///   while this build script runs.
    /// - `CUDA_OXIDE_BACKEND`, `CUDA_OXIDE_LLC`, `CUDA_HOME`: forwarded to
    ///   cargo-oxide as-is (see its docs).
    #[cfg(feature = "cuda-oxide")]
    fn build_ptx_cuda_oxide(&self, output_dir: &Path) {
        println!("cargo:rerun-if-env-changed=CUDA_OXIDE_TARGET");
        println!("cargo:rerun-if-env-changed=CUDA_OXIDE_BACKEND");
        println!("cargo:rerun-if-env-changed=KHAL_CUDA_OXIDE_TARGET_DIR");

        // Keep every intermediate (.ll/.ptx, cargo target dir) OUT of
        // `output_dir`: consumers embed that whole directory with
        // `include_dir!`, so anything left there ends up in the binary.
        let target_dir = std::env::var_os("KHAL_CUDA_OXIDE_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| cargo_target_dir_from_out_dir().join("khal-cuda-oxide"));
        let ptx_dir = target_dir.join("ptx");
        std::fs::create_dir_all(&ptx_dir)
            .unwrap_or_else(|e| panic!("failed to create cuda-oxide PTX dir {ptx_dir:?}: {e}"));

        let mut features = self.features.clone();
        if !features.iter().any(|f| f == "cuda-oxide") {
            features.push("cuda-oxide".to_string());
        }
        let features_str = features.join(",");

        // `cargo oxide build [opts] -- <cargo args>` is cargo-oxide's
        // "passthrough" mode: it runs `cargo build <cargo args>` in the current
        // directory (our shader crate) with the cuda-oxide codegen backend
        // injected, without touching sources or cleaning artifacts.
        let mut cmd = Command::new("cargo");
        cmd.args([
            "oxide",
            "build",
            "--features",
            &features_str,
            "--cargo-target-dir",
        ])
        .arg(&target_dir);

        // cargo-oxide reads CUDA_OXIDE_TARGET itself; only pass `--arch` when
        // the target comes from elsewhere (KHAL_CUDA_ARCH or GPU detection).
        let arch = resolve_sm_arch();
        if std::env::var_os("CUDA_OXIDE_TARGET").is_none()
            && let Some(arch) = &arch
        {
            cmd.args(["--arch", arch]);
        }

        cmd.args(["--", "--release"])
            .current_dir(&self.shader_crate)
            .env("CUDA_OXIDE_PTX_DIR", &ptx_dir)
            .env("RUST_MIN_STACK", self.rust_min_stack.to_string());

        println!(
            "cargo:warning=khal-builder: compiling {} to PTX with cargo-oxide (target {})",
            self.shader_crate.display(),
            arch.as_deref().unwrap_or("cuda-oxide default")
        );

        let status = cmd.status().unwrap_or_else(|e| {
            panic!(
                "failed to run `cargo oxide` ({e}). Install it with \
                 `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide` \
                 using the nightly pinned by cuda-oxide, or set \
                 CUDA_OXIDE_SHADERS_PTX_<SHADER_CRATE> to a prebuilt PTX/cubin."
            )
        });
        if !status.success() {
            panic!("cargo oxide build failed");
        }

        let ptx = find_cuda_oxide_ptx(&ptx_dir, &self.shader_crate);
        let dst = output_dir.join("shaders.ptx");
        std::fs::copy(&ptx, &dst)
            .unwrap_or_else(|e| panic!("failed to copy {ptx:?} to {dst:?}: {e}"));
        println!("cargo:rerun-if-changed={}", ptx.display());
    }
}

/// The cargo target directory of the build that is running this build
/// script, derived from `OUT_DIR` (`<target>/<profile>/build/<pkg>-<hash>/out`).
/// Falls back to a directory under `OUT_DIR` when the layout is unexpected.
#[cfg(feature = "cuda-oxide")]
fn cargo_target_dir_from_out_dir() -> PathBuf {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let looks_like_cargo_layout = out_dir
        .ancestors()
        .nth(2)
        .is_some_and(|b| b.ends_with("build"));
    out_dir
        .ancestors()
        .nth(4)
        .filter(|_| looks_like_cargo_layout)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| out_dir.join("target"))
}

/// The `sm_XY` target for PTX/cubin generation: `KHAL_CUDA_ARCH`, else
/// `CUDA_OXIDE_TARGET`, else the local GPU's compute capability.
#[cfg(feature = "cuda")]
fn resolve_sm_arch() -> Option<String> {
    println!("cargo:rerun-if-env-changed=KHAL_CUDA_ARCH");
    std::env::var("KHAL_CUDA_ARCH")
        .or_else(|_| std::env::var("CUDA_OXIDE_TARGET"))
        .ok()
        .filter(|a| !a.is_empty())
        .or_else(detect_local_sm_arch)
}

/// Assembles `<output_dir>/shaders.ptx` in place into a cubin for the local
/// GPU with the CUDA toolkit's `ptxas`, when possible.
///
/// Why: the CUDA driver JIT-compiles PTX text at load time and rejects PTX
/// whose ISA version is newer than the driver supports
/// (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, e.g. PTX 9.3 from a 13.3 toolkit on
/// a driver that only supports CUDA 13.2). A cubin assembled by the toolkit's
/// own `ptxas` loads regardless of that version skew, and skips the JIT at
/// startup. khal's CUDA loader detects cubins by their ELF magic, so the file
/// keeps its `shaders.ptx` name.
///
/// Best effort: when `ptxas` or the target arch cannot be determined, or
/// `ptxas` fails, the PTX text is left as-is (it may still JIT fine). Set
/// `KHAL_CUDA_KEEP_PTX=1` to always keep forward-compatible PTX text.
#[cfg(feature = "cuda")]
fn assemble_cubin(ptx_path: &Path) {
    println!("cargo:rerun-if-env-changed=KHAL_CUDA_KEEP_PTX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if std::env::var_os("KHAL_CUDA_KEEP_PTX").is_some() {
        return;
    }
    // Already a cubin (e.g. a prebuilt one supplied via CUDA_OXIDE_SHADERS_PTX_*).
    match std::fs::read(ptx_path) {
        Ok(bytes) if bytes.starts_with(b"\x7fELF") => return,
        Ok(_) => {}
        Err(e) => panic!("cannot read {ptx_path:?}: {e}"),
    }

    let Some(arch) = resolve_sm_arch() else {
        println!(
            "cargo:warning=khal-builder: no GPU arch known (set KHAL_CUDA_ARCH=sm_XX); \
             keeping PTX text for driver JIT"
        );
        return;
    };
    let Some(ptxas) = find_cuda_tool("ptxas") else {
        println!(
            "cargo:warning=khal-builder: ptxas not found (PATH, CUDA_HOME, CUDA_PATH, \
             /usr/local/cuda); keeping PTX text for driver JIT"
        );
        return;
    };

    let cubin_path = ptx_path.with_extension("cubin.tmp");
    let status = Command::new(&ptxas)
        .args(["-arch", &arch, "-O3"])
        .arg(ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .status();
    match status {
        Ok(st) if st.success() => {
            std::fs::rename(&cubin_path, ptx_path)
                .unwrap_or_else(|e| panic!("failed to move {cubin_path:?} over {ptx_path:?}: {e}"));
            println!(
                "cargo:warning=khal-builder: assembled {} to a {arch} cubin with {}",
                ptx_path.display(),
                ptxas.display()
            );
        }
        Ok(st) => {
            let _ = std::fs::remove_file(&cubin_path);
            println!(
                "cargo:warning=khal-builder: ptxas exited with {st}; keeping PTX text for driver JIT"
            );
        }
        Err(e) => {
            let _ = std::fs::remove_file(&cubin_path);
            println!(
                "cargo:warning=khal-builder: failed to run {}: {e}; keeping PTX text for driver JIT",
                ptxas.display()
            );
        }
    }
}

/// A CUDA toolkit binary: from `PATH`, else `$CUDA_HOME/bin`, `$CUDA_PATH/bin`,
/// `/usr/local/cuda/bin`.
#[cfg(feature = "cuda")]
fn find_cuda_tool(name: &str) -> Option<PathBuf> {
    if Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return Some(PathBuf::from(name));
    }
    ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(|var| std::env::var_os(var))
        .map(PathBuf::from)
        .chain(std::iter::once(PathBuf::from("/usr/local/cuda")))
        .map(|root| root.join("bin").join(name))
        .find(|p| p.is_file())
}

/// `sm_XY` of the first local GPU as reported by `nvidia-smi`, if available.
#[cfg(feature = "cuda")]
fn detect_local_sm_arch() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let cap = String::from_utf8_lossy(&out.stdout);
    let cap = cap.lines().next()?.trim();
    let (major, minor) = cap.split_once('.')?;
    let major: u32 = major.parse().ok()?;
    let minor: u32 = minor.parse().ok()?;
    Some(format!("sm_{major}{minor}"))
}

/// The `.ptx` cargo-oxide produced for the shader crate: `<package>.ptx`
/// (hyphens normalised to underscores), or the only `.ptx` in the directory.
#[cfg(feature = "cuda-oxide")]
fn find_cuda_oxide_ptx(ptx_dir: &Path, shader_crate: &Path) -> PathBuf {
    let package_name = std::fs::read_to_string(shader_crate.join("Cargo.toml"))
        .ok()
        .and_then(|manifest| {
            manifest.lines().find_map(|line| {
                let line = line.trim();
                let value = line.strip_prefix("name")?.trim_start().strip_prefix('=')?;
                Some(value.trim().trim_matches('"').replace('-', "_"))
            })
        })
        .or_else(|| {
            shader_crate
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.replace('-', "_"))
        });

    if let Some(name) = &package_name {
        let candidate = ptx_dir.join(format!("{name}.ptx"));
        if candidate.is_file() {
            return candidate;
        }
    }

    let mut ptx_files: Vec<PathBuf> = std::fs::read_dir(ptx_dir)
        .unwrap_or_else(|e| panic!("failed to read {ptx_dir:?}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "ptx"))
        .collect();
    match ptx_files.len() {
        1 => ptx_files.remove(0),
        0 => panic!(
            "cargo oxide produced no .ptx in {ptx_dir:?}. Does the shader crate enable its \
             `cuda-oxide` feature and contain `#[spirv_bindgen]` kernels?"
        ),
        _ => panic!(
            "cargo oxide produced several .ptx files in {ptx_dir:?} ({ptx_files:?}) and none is \
             named after the shader package {package_name:?}"
        ),
    }
}
