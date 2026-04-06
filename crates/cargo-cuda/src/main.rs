//! `cargo cuda` — compile Rust shader crates to PTX using the Rust-CUDA toolchain.
//!
//! Follows the same architecture as `cargo gpu`:
//! 1. `cargo cuda install` — builds `rustc_codegen_nvvm` via a dummy crate, caches the DLL
//! 2. `cargo cuda build` — invokes `cargo` with `-Zcodegen-backend=<cached DLL>` on the shader crate

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The nightly toolchain version required by Rust-CUDA's `rustc_codegen_nvvm`.
const RUST_CUDA_NIGHTLY: &str = "nightly-2025-08-04";

/// The Rust-CUDA git repository.
const RUST_CUDA_REPO: &str = "https://github.com/rust-gpu/rust-cuda";

/// The Rust-CUDA git revision.
const RUST_CUDA_REV: &str = "2479a9c77001e58d45beb0f972350fb89181a48e";

#[derive(Parser)]
#[command(name = "cargo-cuda", bin_name = "cargo cuda")]
#[command(about = "Compile Rust shader crates to PTX using the Rust-CUDA toolchain")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build `rustc_codegen_nvvm` and install the required nightly toolchain.
    Install {
        /// Force rebuild even if the codegen backend is already cached.
        #[arg(long)]
        rebuild: bool,
    },
    /// Compile a shader crate to PTX.
    Build(BuildArgs),
}

#[derive(Parser)]
struct BuildArgs {
    /// Path to the shader crate to compile.
    #[arg(long, default_value = ".")]
    shader_crate: PathBuf,

    /// Directory to write the output `.ptx` file(s) to.
    #[arg(long)]
    output_dir: PathBuf,

    /// Comma-separated list of features to enable on the shader crate.
    #[arg(long)]
    features: Option<String>,
}

fn main() {
    let args: Vec<String> = std::env::args()
        .enumerate()
        .filter(|(i, arg)| !(*i == 1 && arg == "cuda"))
        .map(|(_, arg)| arg)
        .collect();

    let cli = Cli::parse_from(args);

    match cli.command {
        Commands::Install { rebuild } => install(rebuild),
        Commands::Build(args) => build(args),
    }
}

// =============================================================================
// Paths
// =============================================================================

/// Returns the cache directory for cargo-cuda artifacts.
fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rust-cuda")
}

/// Returns the directory where the codegen backend is cached.
fn codegen_dir() -> PathBuf {
    cache_dir().join(format!("codegen-{}", &RUST_CUDA_REV[..12]))
}

/// Returns the expected path to the cached codegen backend DLL.
fn codegen_dylib_path() -> PathBuf {
    let name = if cfg!(windows) {
        "rustc_codegen_nvvm.dll"
    } else if cfg!(target_os = "macos") {
        "librustc_codegen_nvvm.dylib"
    } else {
        "librustc_codegen_nvvm.so"
    };
    codegen_dir().join(name)
}

/// Strip the `\\?\` extended-length prefix from Windows canonicalized paths.
fn clean_path(p: &Path) -> String {
    let s = p.display().to_string();
    let s = s.strip_prefix(r"\\?\").unwrap_or(&s);
    s.replace('\\', "/")
}

// =============================================================================
// Install
// =============================================================================

fn install(rebuild: bool) {
    let dylib = codegen_dylib_path();
    if dylib.exists() && !rebuild {
        println!("Codegen backend already cached at {}", dylib.display());
        println!("Use --rebuild to force a rebuild.");
        return;
    }

    // 1. Install the nightly toolchain + components.
    println!("Installing Rust toolchain {RUST_CUDA_NIGHTLY}...");
    run_or_exit(
        Command::new("rustup").args([
            "toolchain",
            "install",
            RUST_CUDA_NIGHTLY,
            "--component",
            "llvm-tools-preview",
            "--component",
            "rust-src",
            "--component",
            "rustc-dev",
        ]),
        "Failed to install toolchain",
    );
    run_or_exit(
        Command::new("rustup").args([
            "target",
            "add",
            "nvptx64-nvidia-cuda",
            "--toolchain",
            RUST_CUDA_NIGHTLY,
        ]),
        "Failed to add nvptx64-nvidia-cuda target",
    );

    // 2. Create a dummy crate to compile rustc_codegen_nvvm.
    let install_dir = codegen_dir();
    std::fs::create_dir_all(install_dir.join("src")).unwrap();

    std::fs::write(
        install_dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "rustc_codegen_nvvm_dummy"
version = "0.1.0"
edition = "2021"

[dependencies.nvvm_codegen]
package = "rustc_codegen_nvvm"
git = "{RUST_CUDA_REPO}"
rev = "{RUST_CUDA_REV}"
"#
        ),
    )
    .unwrap();

    std::fs::write(install_dir.join("src/lib.rs"), "").unwrap();

    std::fs::write(
        install_dir.join("rust-toolchain.toml"),
        format!(
            r#"[toolchain]
channel = "{RUST_CUDA_NIGHTLY}"
components = ["llvm-tools-preview", "rust-src", "rustc-dev"]
"#
        ),
    )
    .unwrap();

    // 3. Build the codegen backend.
    println!("Building rustc_codegen_nvvm (this may take a few minutes)...");

    let nvvm_path = nvvm_path_env();

    let status = Command::new("cargo")
        .args([&format!("+{RUST_CUDA_NIGHTLY}"), "build", "--release"])
        .current_dir(&install_dir)
        .env_remove("RUSTUP_TOOLCHAIN")
        .env("PATH", &nvvm_path)
        .status()
        .expect("failed to run cargo build for codegen backend");

    if !status.success() {
        eprintln!("Failed to build rustc_codegen_nvvm");
        std::process::exit(1);
    }

    // 4. Copy the built DLL to the cache directory root.
    let built_dylib = install_dir
        .join("target/release")
        .join(dylib.file_name().unwrap());
    if built_dylib.exists() {
        std::fs::copy(&built_dylib, &dylib).unwrap();
    } else {
        eprintln!("Codegen backend DLL not found at {}", built_dylib.display());
        std::process::exit(1);
    }

    // 5. Clean up the target dir to save disk space (~200MB).
    let _ = std::fs::remove_dir_all(install_dir.join("target"));

    println!("Codegen backend installed at {}", dylib.display());
}

// =============================================================================
// Build
// =============================================================================

fn build(args: BuildArgs) {
    let dylib = codegen_dylib_path();
    if !dylib.exists() {
        eprintln!("Codegen backend not found. Run `cargo cuda install` first.");
        std::process::exit(1);
    }

    let shader_crate = args.shader_crate.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "Shader crate '{}' not found: {e}",
            args.shader_crate.display()
        );
        std::process::exit(1);
    });

    std::fs::create_dir_all(&args.output_dir).ok();
    let output_dir = args.output_dir.canonicalize().unwrap();

    let dylib_path = clean_path(&dylib);

    println!(
        "Compiling {} to PTX...",
        shader_crate.file_name().unwrap().to_string_lossy()
    );

    let manifest_path = shader_crate.join("Cargo.toml");

    // Use a separate target directory to avoid deadlocking on the workspace
    // cargo lock when invoked from a build script.
    let cuda_target_dir = find_workspace_target_dir(&shader_crate).join("cuda-ptx");

    // Use `rustup run` instead of `cargo +toolchain` because the latter relies
    // on the rustup proxy. When invoked from a build script, `cargo` in PATH
    // may resolve to the real (stable) cargo binary, not the proxy.
    let mut cmd = Command::new("rustup");
    cmd.args(["run", RUST_CUDA_NIGHTLY, "cargo"]);
    // Use `cargo rustc --crate-type cdylib` to override the crate-type for this
    // invocation only, avoiding any modification to the shader crate's Cargo.toml.
    cmd.args(["rustc", "--lib", "--crate-type", "cdylib"]);
    cmd.arg("--manifest-path");
    cmd.arg(&manifest_path);
    cmd.arg("--target-dir");
    cmd.arg(&cuda_target_dir);
    cmd.args([
        "--target",
        "nvptx64-nvidia-cuda",
        "--release",
        "-Zbuild-std=core,alloc",
        "-Zbuild-std-features=panic_immediate_abort",
    ]);

    if let Some(ref features) = args.features {
        cmd.args(["--features", features]);
    }

    // Pass the codegen backend and nvvm options via CARGO_ENCODED_RUSTFLAGS.
    let codegen_flag = format!("-Zcodegen-backend={dylib_path}");
    let nvvm_flags = [
        &codegen_flag,
        "-Zcrate-attr=feature(register_tool)",
        "-Zcrate-attr=register_tool(nvvm_internal)",
        "-Zcrate-attr=no_std",
        "-Zsaturating_float_casts=false",
        "-Cllvm-args=--override-libm",
    ];
    let rustflags = nvvm_flags.join("\x1f"); // Use unit separator for CARGO_ENCODED_RUSTFLAGS

    cmd.env("CARGO_ENCODED_RUSTFLAGS", &rustflags);
    cmd.env_remove("RUSTUP_TOOLCHAIN");
    cmd.env_remove("RUSTFLAGS"); // Don't inherit host RUSTFLAGS
    cmd.env_remove("RUSTC"); // Don't inherit host rustc (e.g. stable from build script)
    cmd.env_remove("RUSTC_WRAPPER");
    cmd.env("PATH", nvvm_path_env());

    let status = cmd.status().expect("failed to run cargo build");

    if !status.success() {
        eprintln!("PTX compilation failed.");
        std::process::exit(1);
    }

    // Find and copy the PTX file(s) from the target directory.
    copy_ptx_files(&shader_crate, &output_dir);
}

/// Finds PTX files in the target directory and copies them to the output directory.
fn copy_ptx_files(shader_crate: &Path, output_dir: &Path) {
    // Read the crate name from Cargo.toml.
    let manifest = shader_crate.join("Cargo.toml");
    let toml = std::fs::read_to_string(&manifest).unwrap();
    let crate_name = toml
        .lines()
        .find_map(|line| {
            let line = line.trim();
            if line.starts_with("name") {
                line.split('=')
                    .nth(1)
                    .map(|v| v.trim().trim_matches('"').replace('-', "_"))
            } else {
                None
            }
        })
        .unwrap_or_else(|| "shaders".to_string());

    // Look for PTX files in the dedicated cuda-ptx target directory.
    let candidates = [
        find_workspace_target_dir(shader_crate).join("cuda-ptx/nvptx64-nvidia-cuda/release/deps")
    ];

    let mut found = false;
    for deps_dir in &candidates {
        if !deps_dir.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(deps_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "ptx")
                    && let Some(stem) = path.file_stem()
                    && stem.to_str().unwrap_or("").starts_with(&crate_name)
                {
                    let dest = output_dir.join("shaders.ptx");
                    std::fs::copy(&path, &dest).unwrap();
                    println!("PTX written to {}", dest.display());
                    found = true;
                    break;
                }
            }
        }
        if found {
            break;
        }
    }

    if !found {
        eprintln!(
            "PTX file not found. Searched in: {}",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        std::process::exit(1);
    }
}

/// Walks up from the crate to find the workspace target directory.
fn find_workspace_target_dir(crate_dir: &Path) -> PathBuf {
    let mut dir = crate_dir.to_path_buf();
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists() {
            let content = std::fs::read_to_string(&manifest).unwrap_or_default();
            if content.contains("[workspace]") {
                return dir.join("target");
            }
        }
        if !dir.pop() {
            break;
        }
    }
    crate_dir.join("target")
}

// =============================================================================
// Helpers
// =============================================================================

/// Returns PATH with NVIDIA's nvvm/bin directory prepended.
fn nvvm_path_env() -> String {
    let mut path = std::env::var("PATH").unwrap_or_default();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let nvvm_bin = format!("{cuda_path}/nvvm/bin");
        if !path.contains(&nvvm_bin) {
            let sep = if cfg!(windows) { ";" } else { ":" };
            path = format!("{nvvm_bin}{sep}{path}");
        }
    }
    path
}

fn run_or_exit(cmd: &mut Command, msg: &str) {
    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("{msg}: {e}");
        std::process::exit(1);
    });
    if !status.success() {
        eprintln!("{msg} (exit code {:?})", status.code());
        std::process::exit(1);
    }
}
