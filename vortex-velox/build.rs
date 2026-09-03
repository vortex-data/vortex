// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::process::exit;

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    for variable in ["MIRI", "MIRIFLAGS", "CARGO_ENCODED_RUSTFLAGS"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if rustflags.contains("sanitizer") || rustflags.contains("address") {
        println!("cargo:info=Skipping header generation under a sanitizer");
        return;
    }
    if env::var("MIRI").is_ok() || env::var("MIRIFLAGS").is_ok() {
        println!("cargo:info=Skipping header generation under Miri");
        return;
    }

    let rustc = Command::new("rustc").arg("-V").output();
    let is_nightly = rustc
        .as_ref()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains("nightly"))
        .unwrap_or(false);
    if !is_nightly {
        println!("cargo:info=Skipping header generation outside nightly Rust");
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("cinclude")
        .join("vortex_velox.h");
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap();
    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate();

    match bindings {
        Ok(bindings) => {
            bindings.write_to_file(&output_file);
            if let Ok(status) = Command::new("clang-format")
                .arg("-i")
                .arg("--style=file")
                .arg(&output_file)
                .status()
                && !status.success()
            {
                println!("cargo:warning=clang-format exited with status {status}");
            }
        }
        Err(error) => {
            println!("cargo:error=Failed to generate vortex_velox.h: {error}");
            exit(1);
        }
    }
}
