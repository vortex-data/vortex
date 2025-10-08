// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use fls_gpu_kernel_gen::generate_unpack;
use walkdir::WalkDir;

fn main() -> anyhow::Result<()> {
    let project_name = "fls-gpu-kernel-gen";
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir"));
    let kernels_dir = manifest_dir.join("kernels");
    let generator_dir = manifest_dir.parent().unwrap().join(project_name);

    fs::create_dir_all(&kernels_dir)?;

    // Generate for all bit widths and both features
    generate_unpack::<u8>(&kernels_dir, 32)?;
    generate_unpack::<u16>(&kernels_dir, 32)?;
    generate_unpack::<u32>(&kernels_dir, 32)?;
    generate_unpack::<u64>(&kernels_dir, 16)?;

    if !has_nvcc() {
        // Don't run cuda compilation if nvcc is not available.
        return Ok(());
    }

    println!("cargo:rerun-if-changed={}", generator_dir.to_str().unwrap());

    for entry in WalkDir::new(&kernels_dir).into_iter().flatten() {
        if entry.path().extension().is_some_and(|ext| ext == "cu") {
            println!("cargo:rerun-if-changed={}", entry.path().display());
            nvcc_compile_ptx(kernels_dir.as_path(), entry.path())?;
        }
    }

    Ok(())
}

fn nvcc_compile_ptx(kernel_dir: &Path, cu_path: &Path) -> anyhow::Result<()> {
    let res = Command::new("nvcc")
        .arg("-arch=sm_80")
        .arg("--restrict")
        .arg("--ptx")
        .arg("--include-path")
        .arg(kernel_dir)
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
