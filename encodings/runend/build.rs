// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AOT-compile Mojo SIMD kernels (`kernels/decode.mojo`) into a static library and link it.
//!
//! When the Mojo compiler is available the build emits `cargo:rustc-cfg=vortex_mojo` so that
//! the Rust side can gate the FFI bridge behind `#[cfg(vortex_mojo)]`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn find_mojo() -> Option<PathBuf> {
    // Check PATH first
    if let Ok(output) = Command::new("which").arg("mojo").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    // Fallback: $HOME/.local/bin/mojo
    if let Ok(home) = env::var("HOME") {
        let candidate = PathBuf::from(home).join(".local/bin/mojo");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/decode.mojo");

    let Some(mojo) = find_mojo() else {
        return;
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let mcpu = env::var("MOJO_MCPU").unwrap_or_else(|_| "native".to_string());

    let obj_path = out_dir.join("decode.o");
    let lib_path = out_dir.join("libvortex_mojo_runend.a");

    // Compile Mojo source to object file
    let Ok(status) = Command::new(&mojo)
        .arg("build")
        .arg("kernels/decode.mojo")
        .arg("--emit")
        .arg("object")
        .arg("--mcpu")
        .arg(&mcpu)
        .arg("--mtune")
        .arg(&mcpu)
        .arg("--target-triple")
        .arg(&target)
        .arg("-o")
        .arg(&obj_path)
        .status()
    else {
        println!("cargo:warning=Mojo compiler failed to launch, falling back to Rust kernels");
        return;
    };

    if !status.success() {
        println!("cargo:warning=Mojo compilation failed ({status}), falling back to Rust kernels");
        return;
    }

    // Archive into a static library
    let Ok(ar_status) = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
    else {
        println!("cargo:warning=ar failed to launch, falling back to Rust kernels");
        return;
    };

    if !ar_status.success() {
        println!("cargo:warning=ar archiving failed, falling back to Rust kernels");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vortex_mojo_runend");
    println!("cargo:rustc-cfg=vortex_mojo");
}
