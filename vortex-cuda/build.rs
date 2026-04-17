// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::panic)]
#![expect(clippy::use_debug)]

use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use fastlanes::FastLanes;

use crate::bit_unpack_gen::generate_cuda_unpack_kernels;
use crate::bit_unpack_gen::generate_cuda_unpack_lanes;

#[path = "src/bit_unpack_gen.rs"]
pub mod bit_unpack_gen;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir");
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    let profile = env::var("PROFILE").unwrap();

    // Source directory for kernels (hand-written and generated .cu/.cuh files)
    let kernels_src = Path::new(&manifest_dir).join("kernels/src");
    // Output directory for compiled .ptx files - separate by profile.
    let kernels_gen = Path::new(&manifest_dir).join("kernels/gen").join(&profile);

    fs::create_dir_all(&kernels_gen).expect("Failed to create kernels/gen directory");

    // Emit the kernels output directory path as a compile-time env var so any binary
    // linking against vortex-cuda can find the PTX files.
    // At runtime, VORTEX_CUDA_KERNELS_DIR can be set to override this path.
    println!(
        "cargo:rustc-env=VORTEX_CUDA_KERNELS_DIR={}",
        kernels_gen.display()
    );

    println!("cargo:rerun-if-env-changed=PROFILE");

    // Regenerate bit_unpack kernels only when the generator changes.
    println!(
        "cargo:rerun-if-changed={}",
        Path::new(&manifest_dir)
            .join("src/bit_unpack_gen.rs")
            .display()
    );
    generate_unpack::<u8>(&kernels_src, 32).expect("Failed to generate unpack for u8");
    generate_unpack::<u16>(&kernels_src, 32).expect("Failed to generate unpack for u16");
    generate_unpack::<u32>(&kernels_src, 32).expect("Failed to generate unpack for u32");
    generate_unpack::<u64>(&kernels_src, 16).expect("Failed to generate unpack for u64");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    generate_dynamic_dispatch_bindings(&kernels_src, &out_dir);
    generate_patches_bindings(&kernels_src, &out_dir);

    if !is_cuda_available() {
        return;
    }

    // Compile .cu files to PTX. We deliberately do NOT register .cu/.cuh files
    // with rerun-if-changed so that editing a .cu file does not trigger Rust
    // recompilation.
    let mut cu_files = Vec::new();
    let mut newest_header = std::time::SystemTime::UNIX_EPOCH;

    if let Ok(entries) = fs::read_dir(&kernels_src) {
        for path in entries.flatten().map(|entry| entry.path()) {
            match path.extension().and_then(|e| e.to_str()) {
                Some("cuh") | Some("h") => {
                    if let Ok(mtime) = fs::metadata(&path).and_then(|m| m.modified()) {
                        newest_header = newest_header.max(mtime);
                    }
                }
                Some("cu") => {
                    cu_files.push(path);
                }
                _ => {}
            }
        }
    }

    // Only compile .cu files whose PTX is stale (older than the source or any header).
    for cu_path in &cu_files {
        let ptx_path = kernels_gen
            .join(cu_path.file_name().unwrap())
            .with_extension("ptx");

        let cu_mtime = fs::metadata(cu_path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let newest_input = cu_mtime.max(newest_header);

        let ptx_mtime = fs::metadata(&ptx_path).and_then(|m| m.modified()).ok();
        if ptx_mtime.is_some_and(|t| t >= newest_input) {
            continue;
        }

        nvcc_compile_ptx(&kernels_src, &kernels_gen, cu_path, &profile)
            .map_err(|e| format!("Failed to compile CUDA kernel {}: {}", cu_path.display(), e))
            .unwrap();
    }
}

fn generate_unpack<T: FastLanes>(output_dir: &Path, thread_count: usize) -> io::Result<()> {
    // Generate the lanes header (.cuh) — device functions only, no __global__ kernels.
    // This is what dynamic_dispatch.cu includes (via bit_unpack.cuh).
    let cuh_path = output_dir.join(format!("bit_unpack_{}_lanes.cuh", T::T));
    let mut cuh_buf = Vec::new();
    generate_cuda_unpack_lanes::<T>(&mut cuh_buf)?;
    write_if_changed(&cuh_path, &cuh_buf);

    // Generate the standalone kernels (.cu) — includes the lanes header,
    // adds _device template + __global__ wrappers. Compiled to its own PTX.
    let cu_path = output_dir.join(format!("bit_unpack_{}.cu", T::T));
    let mut cu_buf = Vec::new();
    generate_cuda_unpack_kernels::<T>(&mut cu_buf, thread_count)?;
    write_if_changed(&cu_path, &cu_buf);

    Ok(())
}

fn write_if_changed(path: &Path, content: &[u8]) {
    if fs::read(path).is_ok_and(|existing| existing == content) {
        return;
    }
    fs::write(path, content).unwrap_or_else(|e| panic!("Failed to write {}: {e}", path.display()));
}

fn nvcc_compile_ptx(
    include_dir: &Path,
    output_dir: &Path,
    cu_path: &Path,
    profile: &str,
) -> io::Result<()> {
    let mut cmd = Command::new("nvcc");
    if profile == "debug" {
        cmd.arg("-O0");

        // NVCC debugging options:
        // https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#debugging-options

        // Include debug symbols for host code.
        cmd.arg("-g");

        // Include debug symbols for device code.
        cmd.arg("-G");

        // Generate line-number information for device code. This option does
        // not affect execution performance and is useful in conjunction with
        // the compute-sanitizer tool to trace the kernel execution.
        cmd.arg("-lineinfo");

        // CUDA Sanitizers
        // - memory: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#using-memcheck
        // - thread: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#using-racecheck
        // - init: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#using-initcheck
        // - synchronize: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#using-synccheck
    } else {
        cmd.arg("-O3");
    }

    // Output PTX file goes to output_dir with same base name
    let ptx_path = output_dir
        .join(cu_path.file_name().unwrap())
        .with_extension("ptx");

    cmd.arg("-std=c++20")
        .arg("-arch=native")
        // Flags forwarded to Clang.
        .arg("--compiler-options=-Wall -Wextra -Wpedantic -Werror")
        .arg("--restrict")
        .arg("--ptx")
        .arg("--include-path")
        .arg(include_dir)
        .arg("-c")
        .arg(cu_path)
        .arg("-o")
        .arg(&ptx_path);

    let res = cmd.output()?;

    if !res.status.success() {
        let stderr = String::from_utf8_lossy(&res.stderr);
        let stdout = String::from_utf8_lossy(&res.stdout);

        println!(
            "cargo:warning=Failed to compile CUDA kernel: {}",
            cu_path.display()
        );
        println!("cargo:warning=Command: {:?}", cmd);

        if !stdout.is_empty() {
            for line in stdout.lines() {
                println!("cargo:warning=stdout: {}", line);
            }
        }
        if !stderr.is_empty() {
            for line in stderr.lines() {
                println!("cargo:warning=stderr: {}", line);
            }
        }

        return Err(io::Error::other(format!(
            "nvcc compilation failed for {}",
            cu_path.display()
        )));
    }
    Ok(())
}

/// Generate bindings for the dynamic dispatch shared header.
fn generate_dynamic_dispatch_bindings(kernels_src: &Path, out_dir: &Path) {
    let header = kernels_src.join("dynamic_dispatch.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .derive_copy(true)
        .derive_debug(true)
        .generate()
        .expect("Failed to generate dynamic_dispatch bindings");

    bindings
        .write_to_file(out_dir.join("dynamic_dispatch.rs"))
        .expect("Failed to write dynamic_dispatch.rs");
}

/// Generate bindings for patches shared header.
fn generate_patches_bindings(kernels_src: &Path, out_dir: &Path) {
    let header = kernels_src.join("patches.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .derive_copy(true)
        .derive_debug(true)
        .generate()
        .expect("Failed to generate patches bindings");

    bindings
        .write_to_file(out_dir.join("patches.rs"))
        .expect("Failed to write patches.rs");
}

fn is_cuda_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}
