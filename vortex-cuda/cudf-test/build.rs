// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = manifest_dir.join("cpp");

    // Create build directory
    let build_dir = out_dir.join("cmake_build");
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Get conda prefix for finding cudf
    let conda_prefix = env::var("CONDA_PREFIX").ok();

    // Configure CMake
    let mut cmake_cmd = Command::new("cmake");
    cmake_cmd
        .current_dir(&build_dir)
        .arg(&cpp_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE=Release"));

    // Add conda prefix to CMAKE_PREFIX_PATH if available
    if let Some(prefix) = &conda_prefix {
        cmake_cmd.arg(format!("-DCMAKE_PREFIX_PATH={}", prefix));
    }

    let status = cmake_cmd
        .status()
        .expect("Failed to run cmake configure");

    if !status.success() {
        panic!("CMake configure failed");
    }

    // Build
    let status = Command::new("cmake")
        .current_dir(&build_dir)
        .args(["--build", ".", "--config", "Release", "-j"])
        .status()
        .expect("Failed to run cmake build");

    if !status.success() {
        panic!("CMake build failed");
    }

    // Tell cargo where to find the library
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudf_arrow_ffi");

    // Also link to cudf and its dependencies
    if let Some(prefix) = &conda_prefix {
        println!("cargo:rustc-link-search=native={}/lib", prefix);
    }

    // Rebuild if C++ sources change
    println!("cargo:rerun-if-changed=cpp/cudf_arrow_ffi.cpp");
    println!("cargo:rerun-if-changed=cpp/cudf_arrow_ffi.h");
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");

    // Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header(cpp_dir.join("cudf_arrow_ffi.h").to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("cudf_.*")
        .allowlist_type("CudfResult")
        .allowlist_type("CudfErrorCode")
        .allowlist_type("ArrowSchema")
        .allowlist_type("ArrowArray")
        .allowlist_type("ArrowDeviceArray")
        .allowlist_type("ArrowDeviceType")
        .allowlist_var("ARROW_DEVICE_.*")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
