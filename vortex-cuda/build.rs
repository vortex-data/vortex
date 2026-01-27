// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::use_debug)]

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    if !is_cuda_available() {
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir");
    let kernels_dir = Path::new(&manifest_dir).join("kernels");

    println!("cargo:rerun-if-changed={}", kernels_dir.to_str().unwrap());

    if let Ok(entries) = std::fs::read_dir(&kernels_dir) {
        for path in entries.flatten().map(|entry| entry.path()) {
            match path.extension().and_then(|e| e.to_str()) {
                Some("cuh") => println!("cargo:rerun-if-changed={}", path.display()),
                Some("cu") => {
                    println!("cargo:rerun-if-changed={}", path.display());
                    // Compile .cu files to PTX
                    nvcc_compile_ptx(&kernels_dir, &path)
                        .map_err(|e| {
                            format!("Failed to compile CUDA kernel {}: {}", path.display(), e)
                        })
                        .unwrap();
                }
                _ => {}
            }
        }
    }
}

fn nvcc_compile_ptx(kernel_dir: &Path, cu_path: &Path) -> std::io::Result<()> {
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    let profile = env::var("PROFILE").unwrap();

    let mut cmd = Command::new("nvcc");
    if profile.as_str() == "debug" {
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

    cmd.arg("-std=c++17")
        .arg("-arch=native")
        // Flags forwarded to Clang.
        .arg("--compiler-options=-Wall -Wextra -Wpedantic -Werror")
        .arg("--restrict")
        .arg("--ptx")
        .arg("--include-path")
        .arg(kernel_dir)
        .arg("-c")
        .arg(cu_path)
        .arg("-o")
        .arg(cu_path.with_extension("ptx"));

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

        return Err(std::io::Error::other(format!(
            "nvcc compilation failed for {}",
            cu_path.display()
        )));
    }
    Ok(())
}

/// Check if CUDA is available based on nvcc.
fn is_cuda_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}
