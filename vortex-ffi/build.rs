// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
use std::env;
use std::path::PathBuf;

fn main() {
    // Skip header generation under miri (miri fails on cbindgen)
    if env::var("MIRI").is_ok() {
        println!("cargo:warning=Skipping header generation under miri");
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir).join("cinclude").join("vortex.h");

    // Create output directory
    std::fs::create_dir_all(output_file.parent().unwrap()).unwrap();

    // Load and potentially modify config for stable toolchain compatibility
    let mut config = cbindgen::Config::from_file("cbindgen.toml").unwrap();

    // Disable macro expansion on stable toolchain to avoid nightly-only features
    let is_nightly = std::process::Command::new("rustc")
        .arg("-V")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains("nightly"))
        .unwrap_or(false);

    if !is_nightly {
        config.parse.expand = cbindgen::ParseExpandConfig {
            crates: Vec::new(),
            ..Default::default()
        };
    }

    // Generate and write header
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .unwrap()
        .write_to_file(&output_file);

    // Set up dependency tracking
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
}
