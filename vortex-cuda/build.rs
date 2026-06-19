// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::use_debug)]

use std::env;
use std::fs::File;
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
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir"));
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    let profile = env::var("PROFILE").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Source directory for hand-written kernels.
    let kernels_src = manifest_dir.join("kernels/src");
    // Generated CUDA source and PTX are build artifacts and must stay under OUT_DIR.
    let generated_kernels_src = out_dir.join("generated-kernels/src");
    let kernels_gen = out_dir.join("kernels").join(&profile);

    std::fs::create_dir_all(&kernels_gen).expect("Failed to create kernels directory");
    std::fs::create_dir_all(&generated_kernels_src)
        .expect("Failed to create generated kernels directory");

    // Always emit the kernels output directory path as a compile-time env var so any binary
    // linking against vortex-cuda can find the PTX files. This must be set regardless
    // of CUDA availability since the code using env!() is always compiled.
    // At runtime, VORTEX_CUDA_KERNELS_DIR can be set to override this path.
    println!(
        "cargo:rustc-env=VORTEX_CUDA_KERNELS_DIR={}",
        kernels_gen.display()
    );

    println!("cargo:rerun-if-env-changed=PROFILE");

    // Regenerate bit_unpack kernels only when the generator changes
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/bit_unpack_gen.rs").display()
    );
    let generated_cuda_sources = [
        generate_unpack::<u8>(&generated_kernels_src, 32)
            .expect("Failed to generate unpack for u8"),
        generate_unpack::<u16>(&generated_kernels_src, 32)
            .expect("Failed to generate unpack for u16"),
        generate_unpack::<u32>(&generated_kernels_src, 32)
            .expect("Failed to generate unpack for u32"),
        generate_unpack::<u64>(&generated_kernels_src, 16)
            .expect("Failed to generate unpack for u64"),
    ];

    generate_arrow_device_array_bindings(&manifest_dir, &out_dir);
    generate_dynamic_dispatch_bindings(&kernels_src, &out_dir);
    generate_patches_bindings(&kernels_src, &out_dir);

    if !is_cuda_available() {
        return;
    }

    let include_dirs = [kernels_src.as_path(), generated_kernels_src.as_path()];

    // Watch and compile hand-written .cu and .cuh files from kernels/src.
    if let Ok(entries) = std::fs::read_dir(&kernels_src) {
        for path in entries.flatten().map(|entry| entry.path()) {
            match path.extension().and_then(|e| e.to_str()) {
                Some("cuh") | Some("h") => {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
                Some("cu") => {
                    println!("cargo:rerun-if-changed={}", path.display());
                    nvcc_compile_ptx(&include_dirs, &kernels_gen, &path, &profile)
                        .map_err(|e| {
                            format!("Failed to compile CUDA kernel {}: {}", path.display(), e)
                        })
                        .unwrap();
                }
                _ => {}
            }
        }
    }

    for path in &generated_cuda_sources {
        nvcc_compile_ptx(&include_dirs, &kernels_gen, path, &profile)
            .map_err(|e| format!("Failed to compile CUDA kernel {}: {}", path.display(), e))
            .unwrap();
    }
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

fn nvcc_compile_ptx(
    include_dirs: &[&Path; 2],
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
        .arg("--ptx");

    for include_dir in include_dirs {
        cmd.arg("--include-path").arg(include_dir);
    }

    cmd.arg("-c").arg(cu_path).arg("-o").arg(&ptx_path);

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
