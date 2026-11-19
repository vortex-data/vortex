// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
use std::env;
use std::fmt::Write;
use std::path::PathBuf;
use std::process::Command;
use std::process::exit;

use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::declarations::core;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(vortex_asan)");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    for env in ["MIRI", "MIRIFLAGS", "CARGO_ENCODED_RUSTFLAGS"] {
        println!("cargo:rerun-if-env-changed={env}");
    }

    // Skip cbindgen under sanitizers: its nested expansion build drops the sanitizer flag and
    // mismatches the sanitizer-built deps.
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();

    if rustflags.contains("address") {
        println!("cargo:info=building with asan");
        println!("cargo:rustc-cfg=vortex_asan");
        println!("cargo:info=Skipping header generation due to sanitizers");
        return;
    }

    if rustflags.contains("sanitizer") {
        println!("cargo:info=Skipping header generation due to sanitizers");
        return;
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

    let mut config = cbindgen::Config::from_file("cbindgen.toml").unwrap();
    let header = config.header.get_or_insert_default();
    header.push_str("\n// Core edition names accepted by vx_session_enable_edition.\n");
    for declaration in EDITION_DECLARATIONS {
        let id = declaration.edition.id;
        if id.family == core::FAMILY.name {
            writeln!(
                header,
                "#define VX_EDITION_CORE_{:04}_{:02}_{} \"{id}\"",
                id.year, id.month, id.version
            )
            .unwrap();
        }
    }

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
