// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::use_debug)]

use std::env;
use std::fs::File;
use std::io;
use std::io::Write;
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
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    // Keep different architecture selections isolated across build trees.
    let kernels_gen = out_dir.join("kernels");

    // NVCC's output depends on its flags and, with native, the visible GPUs.
    for name in [
        "PROFILE",
        "PATH",
        "CUDA_PATH",
        "VORTEX_CUDA_ARCH_FLAGS",
        "VORTEX_CUDA_HOST_COMPILER",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "NVCC_CCBIN",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    // Regenerate bit_unpack kernels only when the generator changes
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

    generate_arrow_device_array_bindings(Path::new(&manifest_dir), &out_dir);
    generate_dynamic_dispatch_bindings(&kernels_src, &out_dir);
    generate_patches_bindings(&kernels_src, &out_dir);

    if !is_cuda_available() {
        // The kernel loader unconditionally includes embedded_kernels.rs, so emit a Rust stub
        // even without nvcc, replacing any stale table from a previous CUDA-enabled build.
        generate_embedded_kernels(&out_dir, &[]).expect("Failed to generate empty kernel table");
        return;
    }

    std::fs::create_dir_all(&kernels_gen).expect("Failed to create kernel output directory");
    let mut kernel_files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&kernels_src) {
        for path in entries.flatten().map(|entry| entry.path()) {
            let is_generated = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("bit_unpack_"));

            match path.extension().and_then(|e| e.to_str()) {
                // Only watch hand-written .cuh/.h files, not generated ones
                // (generated files are rebuilt when cuda_kernel_generator changes)
                Some("cuh") | Some("h") if !is_generated => {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
                Some("cu") => {
                    // Only watch hand-written .cu files, not generated ones
                    // (generated files are rebuilt when cuda_kernel_generator changes)
                    if !is_generated {
                        println!("cargo:rerun-if-changed={}", path.display());
                    }
                    let kernel = nvcc_compile_kernel(&kernels_src, &kernels_gen, &path, &profile)
                        .map_err(|e| {
                            format!("Failed to compile CUDA kernel {}: {}", path.display(), e)
                        })
                        .unwrap();
                    kernel_files.push(kernel);
                }
                _ => {}
            }
        }
    }

    kernel_files.sort();
    generate_embedded_kernels(&out_dir, &kernel_files)
        .expect("Failed to generate embedded kernels");
}

/// Embeds only this build's fatbins, excluding stale outputs from previous configurations.
fn generate_embedded_kernels(out_dir: &Path, kernel_files: &[PathBuf]) -> io::Result<()> {
    let mut file = File::create(out_dir.join("embedded_kernels.rs"))?;
    if kernel_files.is_empty() {
        writeln!(
            file,
            "pub(crate) fn embedded_kernel(_module_name: &str) -> Option<&'static [u8]> {{ None }}"
        )?;
        return Ok(());
    }
    writeln!(
        file,
        "pub(crate) fn embedded_kernel(module_name: &str) -> Option<&'static [u8]> {{"
    )?;
    writeln!(file, "    match module_name {{")?;
    for path in kernel_files {
        let Some(module_name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        writeln!(
            file,
            "        {module_name:?} => Some(include_bytes!({:?})),",
            path.to_string_lossy()
        )?;
    }
    writeln!(file, "        _ => None,")?;
    writeln!(file, "    }}")?;
    writeln!(file, "}}")?;
    Ok(())
}

fn generate_unpack<T: FastLanes>(output_dir: &Path, thread_count: usize) -> io::Result<PathBuf> {
    // Generate the lanes header (.cuh) — device functions only, no __global__ kernels.
    // This is what dynamic_dispatch.cu includes (via bit_unpack.cuh).
    let cuh_path = output_dir.join(format!("bit_unpack_{}_lanes.cuh", T::T));
    let mut cuh_file = File::create(&cuh_path)?;
    generate_cuda_unpack_lanes::<T>(&mut cuh_file)?;

    // Generate the standalone kernels (.cu) — includes the lanes header,
    // adds _device template + __global__ wrappers. Compiled to its own PTX.
    let cu_path = output_dir.join(format!("bit_unpack_{}.cu", T::T));
    let mut cu_file = File::create(&cu_path)?;
    generate_cuda_unpack_kernels::<T>(&mut cu_file, thread_count)?;

    Ok(cu_path)
}

fn nvcc_compile_kernel(
    include_dir: &Path,
    output_dir: &Path,
    cu_path: &Path,
    profile: &str,
) -> io::Result<PathBuf> {
    let architecture_flags =
        env::var("VORTEX_CUDA_ARCH_FLAGS").unwrap_or_else(|_| "-arch=native".to_owned());
    let mut cmd = Command::new("nvcc");
    cmd.args(architecture_flags.split_whitespace());
    if let Some(host_compiler) =
        env::var_os("VORTEX_CUDA_HOST_COMPILER").filter(|path| !path.is_empty())
    {
        // Despite its name, --compiler-bindir accepts a compiler executable path.
        // It is used to honor CMake's host compiler choice; CXX alone does not reliably control NVCC.
        cmd.arg("--compiler-bindir").arg(host_compiler);
    }
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

    let fatbin_path = output_dir
        .join(cu_path.file_name().unwrap())
        .with_extension("fatbin");

    cmd.arg("-std=c++20")
        // Flags forwarded to Clang.
        .arg("--compiler-options=-Wall -Wextra -Wpedantic -Werror")
        .arg("--restrict")
        // Fatbins preserve real/virtual architecture lists; PTX alone cannot.
        .arg("--fatbin")
        .arg("--include-path")
        .arg(include_dir)
        .arg(cu_path)
        .arg("-o")
        .arg(&fatbin_path);

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
    Ok(fatbin_path)
}

/// Generate bindings for the vendored Arrow C Device ABI header.
fn generate_arrow_device_array_bindings(manifest_dir: &Path, out_dir: &Path) {
    let header = manifest_dir.join("src/arrow/reference/arrow_c_device.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_type("ArrowArray")
        .allowlist_type("ArrowDeviceArray")
        .allowlist_type("ArrowDeviceArrayStream")
        .allowlist_type("ArrowDeviceType")
        .allowlist_var("ARROW_DEVICE_.*")
        // ArrowArray/ArrowDeviceArray own producer state through release/private_data.
        // Shallow copies must use Arrow C move semantics, not Rust Copy/Clone.
        .derive_copy(false)
        .derive_debug(true)
        .layout_tests(false)
        .generate()
        .expect("Failed to generate Arrow C Device bindings");

    bindings
        .write_to_file(out_dir.join("arrow_c_abi.rs"))
        .expect("Failed to write arrow_c_abi.rs");
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
        .expect("Failed to generate dynamic_dispatch bindings");

    bindings
        .write_to_file(out_dir.join("patches.rs"))
        .expect("Failed to write patches.rs");
}

/// Check if CUDA is available based on nvcc.
fn is_cuda_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}
