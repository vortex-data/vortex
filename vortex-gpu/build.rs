// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use walkdir::WalkDir;

fn main() -> anyhow::Result<()> {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir"));
    let kernels_dir = manifest_dir.join("kernels");
    let generator_dir = manifest_dir.parent().unwrap().join("fls-gpu-kernel-gen");
    let fls_gen_output = Command::new("cargo")
        .current_dir(&generator_dir)
        .arg("run")
        .arg("--")
        .arg("--output-dir")
        .arg(&kernels_dir)
        .output()
        .expect("Failed to run fls-gpu-kernel-gen");

    assert!(
        fls_gen_output.status.success(),
        "Failed to run fls-gpu-kernel-gen: {}",
        str::from_utf8(&fls_gen_output.stderr)?
    );

    if !has_nvcc() {
        // Don't run cuda compilation if nvcc is not available.
        return Ok(());
    }

    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");
    println!("cargo:rerun-if-changed={}", generator_dir.to_str().unwrap());

    for entry in WalkDir::new(kernels_dir).into_iter().flatten() {
        if entry.path().extension().is_some_and(|ext| ext == "cu") {
            nvcc_compile_ptx(entry.path())?;
        }
    }

    Ok(())
}

fn nvcc_compile_ptx(cu_path: &Path) -> anyhow::Result<()> {
    let res = Command::new("nvcc")
        .arg("-arch=sm_80")
        .arg("--restrict")
        .arg("--ptx")
        .arg("-c")
        .arg(cu_path)
        .arg("-o")
        .arg(cu_path.with_extension("ptx"))
        .output()?;

    assert!(
        res.status.success(),
        "Failed to compile {}: {}",
        cu_path.display(),
        str::from_utf8(&res.stderr)?
    );
    Ok(())
}

fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}
