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
    /// If the `cuda` feature is enabled and this is `true`, then cuda PTX kernels will be built with cargo-cuda.
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

        if self.build_spirv {
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
    #[cfg(feature = "cuda")]
    fn build_ptx(&self, output_dir: impl AsRef<Path>) {
        let output_dir = output_dir.as_ref();
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
}
