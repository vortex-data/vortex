// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Builds the OnPair C++ library plus a thin C-ABI shim into a static archive
// that gets linked into this crate. The CMake configuration lives in
// `cmake/CMakeLists.txt` and fetches `gargiulofrancesco/onpair_cpp` via
// `FetchContent`.

fn main() {
    let cmake_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cmake");

    println!("cargo:rerun-if-changed={}", cmake_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("cxx")
            .display()
    );
    println!("cargo:rerun-if-env-changed=VORTEX_ONPAIR_FORCE_REBUILD");

    let dst = cmake::Config::new(&cmake_dir)
        .profile("Release")
        .define("CMAKE_POLICY_DEFAULT_CMP0077", "NEW")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("ONPAIR_BUILD_TESTS", "OFF")
        .define("ONPAIR_BUILD_EXAMPLES", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    // The shim depends on onpair; both are static archives.
    println!("cargo:rustc-link-lib=static=onpair_shim");
    println!("cargo:rustc-link-lib=static=onpair");

    // C++ standard library, picked by host platform.
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target.as_str() {
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
        "windows" => {} // MSVC links the runtime automatically.
        _ => println!("cargo:rustc-link-lib=stdc++"),
    }
}
