// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]
use std::env;
use std::path::PathBuf;

fn main() {
    // Set up dependency tracking
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");

    // Skip header generation in environments where cbindgen macro expansion fails
    if env::var("MIRI").is_ok() || env::var("MIRIFLAGS").is_ok() {
        println!("cargo:warning=Skipping header generation under miri (cbindgen incompatible)");
        return;
    }

    // We require the macro expansion feature of cbindgen to generate the header, which is only available on nightly.
    let is_nightly = std::process::Command::new("rustc")
        .arg("-V")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains("nightly"))
        .unwrap_or(false);
    if !is_nightly {
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
            let status = std::process::Command::new("clang-format")
                .arg("-i")
                .arg("--style=file")
                .arg(&output_file)
                .status();
            match status {
                Ok(s) if s.success() => {}
                Ok(s) => println!(
                    "cargo:warning=clang-format exited with status {} for {}",
                    s,
                    output_file.display()
                ),
                Err(e) => println!("cargo:warning=clang-format not found or failed to run: {e}"),
            }
        }
        Err(e) => {
            // Check if this might be a sanitizer-related incompatibility
            let error_msg = e.to_string();
            let rustflags = env::var("RUSTFLAGS").unwrap_or_default();

            if rustflags.contains("sanitizer") || error_msg.contains("sanitizer") {
                println!(
                    "cargo:warning=Skipping header generation due to sanitizer incompatibility"
                );
                println!("cargo:warning=Error: {}", e);
                return;
            }

            // For non-sanitizer errors, fail hard as these indicate real problems
            panic!("Failed to generate header with cbindgen: {}", e);
        }
    }
}
