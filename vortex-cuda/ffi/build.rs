// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::error::Error;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let header = "cinclude/vortex_cuda.h";
    // The CUDA API needs no macro expansion or dependency parsing, so generate on stable Rust
    // without recursively building the CUDA implementation.
    cbindgen::Builder::new()
        .with_src("src/lib.rs")
        .with_config(cbindgen::Config::from_file("cbindgen.toml")?)
        .generate()?
        .write_to_file(header);
    if !Command::new("clang-format")
        .args(["--style=file", "-i"])
        .arg(header)
        .status()
        .is_ok_and(|status| status.success())
    {
        println!("cargo:warning=clang-format unavailable or failed; CUDA header left unformatted");
    }
    Ok(())
}
