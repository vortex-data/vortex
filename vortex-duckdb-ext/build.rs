#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
use std::env;
use std::path::PathBuf;

use walkdir::WalkDir;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Directory of our DuckDB extension.
    let duckdb_ext_dir = crate_dir.parent().unwrap().join("duckdb-vortex");

    // Generate the _imported_ bindings from our C++ code.
    bindgen::Builder::default()
        .header("cpp/include/duckdb_vx.h")
        // Add the #[must_use] attribute to FFI functions that return results.
        .must_use_type("duckdb_state")
        .rustified_enum("duckdb_state")
        .rustified_enum("DUCKDB_VX_EXPR_CLASS")
        .rustified_enum("DUCKDB_VX_EXPR_TYPE")
        .rustified_enum("DUCKDB_VX_TABLE_FILTER_TYPE")
        .rustified_enum("DUCKDB_VX_VECTOR_TYPE")
        .rustified_non_exhaustive_enum("DUCKDB_TYPE")
        .size_t_is_usize(true)
        //.generate_cstr(true) // This emits invalid syntax and breaks the Rust formatter
        .clang_arg(format!(
            "-I{}",
            duckdb_ext_dir.join("duckdb/src/include/").to_str().unwrap()
        ))
        .clang_arg(format!(
            "-I{}",
            crate_dir.join("cpp/include").to_str().unwrap()
        ))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings")
        .write_to_file(crate_dir.join("src/cpp.rs"))
        .expect("Couldn't write bindings!");

    // Compile our C++ code that exposes additional DuckDB functionality.
    cc::Build::new()
        .std("c++17")
        // Allow C++20 designator syntax even with C++17 std
        .flag("-Wno-c++20-designator")
        .flag("-Wno-unused-parameter")
        // We include DuckDB headers from the DuckDB extension submodule.
        .include(duckdb_ext_dir.join("duckdb/src/include"))
        .include("cpp/include")
        .file("cpp/data.cpp")
        .file("cpp/error.cpp")
        .file("cpp/expr.cpp")
        .file("cpp/table_filter.cpp")
        .file("cpp/table_function.cpp")
        .compile("vortex-duckdb-ext-extras");

    // Generate the _exported_ bindings from our Rust code.
    cbindgen::Builder::new()
        .with_config(cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).unwrap())
        .with_crate(&crate_dir)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(crate_dir.join("include/vortex.h"));

    for entry in WalkDir::new("cpp/") {
        println!("cargo:rerun-if-changed={}", entry.unwrap().path().display());
    }
    println!("cargo:rerun-if-changed=src/cpp.rs");
}
