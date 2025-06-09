#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
use std::path::PathBuf;
use std::{env, fs};

const DUCKDB_VERSION: &str = "v1.3.0";
const DUCKDB_BASE_URL: &str = "https://github.com/duckdb/duckdb/releases/download";

fn download_duckdb_archive() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let duckdb_dir = out_dir.join(format!("duckdb-{DUCKDB_VERSION}"));

    let target = env::var("TARGET")?;
    let (platform, arch) = match target.as_str() {
        "aarch64-apple-darwin" => ("osx", "arm64"),
        "x86_64-apple-darwin" => ("osx", "amd64"),
        "x86_64-unknown-linux-gnu" | "x86_64-unknown-linux-musl" => ("linux", "amd64"),
        "aarch64-unknown-linux-gnu" | "aarch64-unknown-linux-musl" => ("linux", "arm64"),
        _ => return Err(format!("Unsupported target: {target}").into()),
    };

    let archive_name = format!("static-libs-{platform}-{arch}.zip");
    let url = format!("{DUCKDB_BASE_URL}/{DUCKDB_VERSION}/{archive_name}");
    let archive_path = duckdb_dir.join(&archive_name);

    // Create directory if it doesn't exist.
    fs::create_dir_all(&duckdb_dir)?;

    if !archive_path.exists() {
        println!("Downloading DuckDB static libraries from {url}");
        let response = reqwest::blocking::get(&url)?;
        fs::write(&archive_path, &response.bytes()?)?;
        println!("Downloaded to {}", archive_path.display());
    } else {
        println!("Archive already exists, skipping download");
    }

    Ok(archive_path)
}

fn extract_duckdb_libraries(archive_path: PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let duckdb_dir = archive_path
        .parent()
        .ok_or("Invalid archive path")?
        .to_path_buf();

    // Check if already extracted.
    if duckdb_dir.join("libduckdb_static.a").exists() {
        println!("DuckDB libraries already extracted, skipping extraction");
        return Ok(duckdb_dir);
    }

    println!("Extracting DuckDB static libraries");
    let file = fs::File::open(&archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    archive.extract(&duckdb_dir)?;

    println!("Extracted DuckDB libraries to {}", duckdb_dir.display());
    Ok(duckdb_dir)
}

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

    // Download and extract prebuilt DuckDB libraries.
    let zip_path = download_duckdb_archive().unwrap();
    let lib_path = extract_duckdb_libraries(zip_path).unwrap();

    // Link against static DuckDB libraries.
    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=static=duckdb_static");
    println!("cargo:rustc-link-lib=static=duckdb_fmt");

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
        .with_no_includes()
        .generate()
        .expect("error: Unable to generate bindings for vortex.h")
        .write_to_file(crate_dir.join("include/vortex.h"));

    for entry in walkdir::WalkDir::new("cpp/") {
        println!("cargo:rerun-if-changed={}", entry.unwrap().path().display());
    }
    println!("cargo:rerun-if-changed=src/cpp.rs");
}
