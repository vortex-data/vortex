// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::process::exit;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(vortex_asan)");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    for env in ["MIRI", "MIRIFLAGS", "CARGO_ENCODED_RUSTFLAGS"] {
        println!("cargo:rerun-if-env-changed={env}");
    }

    let is_asan = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .contains("address");
    if is_asan {
        println!("cargo:info=building with asan");
        println!("cargo:rustc-cfg=vortex_asan");
        println!("cargo:info=Skipping header generation due to sanitizers");
        return;
    }

    if env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .contains("sanitizer")
    {
        println!("cargo:info=Skipping header generation due to sanitizers");
    }

    if env::var("MIRI").is_ok() || env::var("MIRIFLAGS").is_ok() {
        println!("cargo:info=Skipping header generation under miri (cbindgen incompatible)");
        return;
    }

    // cbindgen macro expansion is only available on nightly
    let rustc = Command::new("rustc").arg("-V").output();
    let is_nightly = rustc
        .as_ref()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains("nightly"))
        .unwrap_or(false);
    if !is_nightly {
        println!("cargo:info=Skipping header generation as we're not on nightly");
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir).join("cinclude").join("vortex.h");

    // Create output directory
    std::fs::create_dir_all(output_file.parent().unwrap()).unwrap();

    // Load config
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap();

    // Generate and write header
    let result = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate();

    match result {
        Ok(bindings) => {
            bindings.write_to_file(&output_file);

            // Run clang-format on the generated header.
            if let Ok(status) = Command::new("clang-format")
                .arg("-i")
                .arg("--style=file")
                .arg(&output_file)
                .status()
            {
                if !status.success() {
                    println!("cargo:warning=clang-format exited with status {status}");
                }
            } else {
                println!(
                    "cargo:warning=clang-format not found, skipping formatting of generated header"
                );
            }
        }
        Err(err) => {
            if err.to_string().contains("sanitizer") {
                println!("cargo:info=Skipping header generation due to sanitizers");
                return;
            }
            println!("cargo:error=Failed to generate header with cbindgen: {err}");
            exit(1);
        }
    }
}
