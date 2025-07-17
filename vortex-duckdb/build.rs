// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
use std::path::PathBuf;
use std::{env, fs};

const DUCKDB_VERSION: &str = "1.3.2";
const DUCKDB_BASE_URL: &str = "https://github.com/duckdb/duckdb/releases/download";

fn download_duckdb_lib_archive() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let target_dir = manifest_dir.parent().unwrap().join("target");
    let duckdb_dir = target_dir.join("duckdb-lib");

    let target = env::var("TARGET")?;
    let (platform, arch) = match target.as_str() {
        "aarch64-apple-darwin" => ("osx", "universal"),
        "x86_64-apple-darwin" => ("osx", "universal"),
        "x86_64-unknown-linux-gnu" => ("linux", "amd64"),
        "aarch64-unknown-linux-gnu" => ("linux", "arm64"),
        _ => return Err(format!("Unsupported target: {target}").into()),
    };

    let archive_name = format!("libduckdb-{platform}-{arch}.zip");
    let url = format!("{DUCKDB_BASE_URL}/v{DUCKDB_VERSION}/{archive_name}");
    let archive_path = duckdb_dir.join(&archive_name);

    // Recreate the duckdb directory
    let _ = fs::remove_dir_all(&duckdb_dir);
    fs::create_dir_all(&duckdb_dir)?;

    if !archive_path.exists() {
        println!("Downloading DuckDB libraries from {url}");
        let response = http_client()?.get(&url).send()?;
        fs::write(&archive_path, &response.bytes()?)?;
        println!("Downloaded to {}", archive_path.display());
    } else {
        println!("Archive already exists, skipping download");
    }

    Ok(archive_path)
}

fn extract_duckdb_libraries(archive_path: PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let duckdb_lib_dir = archive_path
        .parent()
        .ok_or("Invalid archive path")?
        .to_path_buf();

    // Check if already extracted. The archive for Linux only contains a .so library, macOS only .dylib.
    if duckdb_lib_dir.join("libduckdb.dylib").exists()
        || duckdb_lib_dir.join("libduckdb.so").exists()
    {
        println!("DuckDB libraries already extracted, skipping extraction");
        return Ok(duckdb_lib_dir);
    }

    let file = fs::File::open(&archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    archive.extract(&duckdb_lib_dir)?;
    println!(
        "Extracting DuckDB libraries to {}",
        duckdb_lib_dir.display()
    );

    Ok(duckdb_lib_dir)
}

fn download_duckdb_source_archive() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let target_dir = manifest_dir.parent().unwrap().join("target");
    let duckdb_source_dir = target_dir.join(format!("duckdb-source-v{DUCKDB_VERSION}"));
    let archive_name = format!("duckdb-source-v{DUCKDB_VERSION}.zip");
    let url = format!("https://github.com/duckdb/duckdb/archive/refs/tags/v{DUCKDB_VERSION}.zip");
    let archive_path = duckdb_source_dir.join(&archive_name);

    // Create directory if it doesn't exist.
    fs::create_dir_all(&duckdb_source_dir)?;

    if !archive_path.exists() {
        println!("Downloading DuckDB source code from {url}");
        let response = http_client()?.get(&url).send()?;
        fs::write(&archive_path, &response.bytes()?)?;
        println!("Downloaded to {}", archive_path.display());
    } else {
        println!("Source archive already exists, skipping download");
    }

    Ok(archive_path)
}

fn extract_duckdb_source(archive_path: PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let duckdb_source_dir = archive_path
        .parent()
        .ok_or("Invalid archive path")?
        .to_path_buf();

    // Check if the source is already extracted.
    if duckdb_source_dir
        .join(format!("duckdb-{DUCKDB_VERSION}/CMakeLists.txt"))
        .exists()
    {
        println!("DuckDB source already extracted, skipping extraction");
        return Ok(duckdb_source_dir);
    }

    let file = fs::File::open(&archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    archive.extract(&duckdb_source_dir)?;
    println!(
        "Extracting DuckDB source to {}",
        duckdb_source_dir.display()
    );

    Ok(duckdb_source_dir)
}

fn http_client() -> Result<reqwest::blocking::Client, Box<dyn std::error::Error>> {
    let timeout_secs = env::var("CARGO_HTTP_TIMEOUT")
        .or_else(|_| env::var("HTTP_TIMEOUT"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(90);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()?;
    Ok(client)
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let duckdb_repo = crate_dir.join("duckdb");

    // Download and extract prebuilt DuckDB libraries.
    let zip_lib_path = download_duckdb_lib_archive().unwrap();
    let extraced_lib_path = extract_duckdb_libraries(zip_lib_path).unwrap();

    // Download, extract and symlink DuckDB source code.
    let zip_source_path = download_duckdb_source_archive().unwrap();
    let extracted_source_path = extract_duckdb_source(zip_source_path).unwrap();
    let _ = fs::remove_dir_all(&duckdb_repo);
    std::os::unix::fs::symlink(&extracted_source_path, &duckdb_repo).unwrap();

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
            duckdb_repo
                .join(format!("duckdb-{DUCKDB_VERSION}/src/include"))
                .to_str()
                .unwrap()
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

    // Link against DuckDB dylib.
    println!(
        "cargo:rustc-link-search=native={}",
        extraced_lib_path.display()
    );
    println!("cargo:rustc-link-lib=dylib=duckdb");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        extraced_lib_path.display()
    );

    if env::var("TARGET").unwrap().contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        println!("cargo:rustc-link-lib=c++");
    }

    // Compile our C++ code that exposes additional DuckDB functionality.
    cc::Build::new()
        .std("c++17")
        // Enable compiler warnings.
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-Wpedantic")
        // Allow C++20 designator syntax even with C++17 std
        .flag("-Wno-c++20-designator")
        // Enable C++20 extensions
        .flag("-Wno-c++20-extensions")
        // Unused parameter warnings are disabled as we include DuckDB
        // headers with implementations that have unused parameters.
        .flag("-Wno-unused-parameter")
        // We include DuckDB headers from the DuckDB extension submodule.
        .include(duckdb_repo.join(format!("duckdb-{DUCKDB_VERSION}/src/include")))
        .include("cpp/include")
        .file("cpp/copy_function.cpp")
        .file("cpp/data.cpp")
        .file("cpp/data_chunk.cpp")
        .file("cpp/error.cpp")
        .file("cpp/expr.cpp")
        .file("cpp/logical_type.cpp")
        .file("cpp/scalar_function.cpp")
        .file("cpp/table_filter.cpp")
        .file("cpp/table_function.cpp")
        .file("cpp/vector.cpp")
        .compile("vortex-duckdb-extras");

    // Generate the _exported_ bindings from our Rust code.
    cbindgen::Builder::new()
        .with_config(cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).unwrap())
        .with_crate(&crate_dir)
        .with_no_includes()
        .generate()
        .expect("error: Unable to generate bindings for vortex.h")
        .write_to_file(crate_dir.join("include/vortex.h"));

    // Watch C/C++ source files for changes.
    for entry in walkdir::WalkDir::new("cpp/").into_iter().flatten() {
        if entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "cpp" || ext == "h" || ext == "hpp")
        {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
}
