// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::process::Command;

fn main() {
    // Declare the cfg so rustc doesn't warn about unexpected cfg.
    println!("cargo::rustc-check-cfg=cfg(cuda_available)");

    // Only enable CUDA on Linux (matching vortex-cuda's behavior)
    if cfg!(not(target_os = "linux")) {
        return;
    }

    // Check if nvcc is available
    if !cuda_available() {
        return;
    }

    println!("cargo:rustc-cfg=cuda_available");
}

fn cuda_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}
