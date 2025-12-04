// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest dir"));
    let mojo_dir = manifest_dir.join("mojo");

    // Check if mojo directory exists
    if !mojo_dir.exists() {
        println!("cargo:warning=Mojo directory not found, skipping Mojo compilation");
        return Ok(());
    }

    // Check if kernel.mojo exists
    let kernel_file = mojo_dir.join("kernel.mojo");
    if !kernel_file.exists() {
        println!("cargo:warning=kernel.mojo not found, skipping Mojo compilation");
        return Ok(());
    }

    println!("cargo:rerun-if-changed={}", kernel_file.display());

    // Check if the macOS library exists
    let lib_path = mojo_dir.join("libvxmojo.dylib");
    if !lib_path.exists() {
        println!(
            "cargo:warning=libvxmojo.dylib not found in {}",
            mojo_dir.display()
        );
        return Ok(());
    }

    // Run pixi shell && mojo build
    // let output = Command::new("bash")
    //     .arg("-c")
    //     .arg("pixi shell && mojo build -O1 --emit shared-lib kernel.mojo -o libvxmojo.dylib")
    //     .current_dir(&mojo_dir)
    //     .output()?;

    // if !output.status.success() {
    //     return Err(anyhow::anyhow!(
    //         "Failed to compile Mojo kernel: {}",
    //         String::from_utf8_lossy(&output.stderr)
    //     ));
    // }

    println!("cargo:rustc-link-search=native={}", mojo_dir.display());
    println!("cargo:rustc-link-lib=dylib=vxmojo");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mojo_dir.display());

    let mojo_runtime_lib = mojo_dir.join(".pixi/envs/default/lib");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        mojo_runtime_lib.display()
    );

    Ok(())
}
