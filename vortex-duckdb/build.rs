// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use std::borrow::ToOwned;
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use bindgen::Abi;
use once_cell::sync::Lazy;

static DUCKDB_VERSION: Lazy<DuckDBVersion> = Lazy::new(|| {
    // Override the DuckDB version via environment variable in case of an extension build.
    // `DUCKDB_VERSION` is set by the extension build in the `duckdb-vortex` repo.
    //
    // This is to ensure that we don't implicitly build against a different DuckDB version during
    // an extension build which might lead to subtle ABI breaks, e.g. reordering fields in C++ structs.
    if let Ok(version) = env::var("DUCKDB_VERSION") {
        // DUCKDB_VERSION env var can be set by:
        // - The extension build in `vortex-data/duckdb-vortex` repo
        // - Developers who want to test against a specific version/commit
        parse_version(&version)
    } else {
        // The default DuckDB version to use when DUCKDB_VERSION env var is not set.
        DuckDBVersion::Release("1.5.0".to_owned())
    }
});

const DUCKDB_RELEASES_URL: &str = "https://github.com/duckdb/duckdb/releases/download";

/// Represents either a released DuckDB version or a specific commit hash.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum DuckDBVersion {
    /// A released DuckDB version of the form "X.Y.Z".
    /// This mode will download pre-compiled dynamic libraries from GitHub releases.
    Release(String),
    /// A commit hash from `github.com/duckdb/duckdb`.
    /// This mode will download source and build DuckDB from scratch.
    Commit(String),
}

impl DuckDBVersion {
    /// Returns the directory name suffix for this version.
    fn dir_suffix(&self) -> String {
        match self {
            DuckDBVersion::Release(v) => format!("v{v}"),
            DuckDBVersion::Commit(c) => c.clone(),
        }
    }

    /// Returns true if this is a release version (not a commit).
    fn is_release(&self) -> bool {
        matches!(self, DuckDBVersion::Release(_))
    }

    /// Returns the name of the extracted source directory inside the zip archive.
    /// GitHub archives extract to `duckdb-{version}` for tags and `duckdb-{commit}` for commits.
    fn archive_inner_dir_name(&self) -> String {
        match self {
            DuckDBVersion::Release(v) => format!("duckdb-{v}"),
            DuckDBVersion::Commit(c) => format!("duckdb-{c}"),
        }
    }
}

/// Parse a version string into a DuckDBVersion.
/// - Strings starting with "v" followed by semver are treated as releases
/// - Pure semver strings (X.Y.Z) are treated as releases
/// - Everything else (e.g., commit hashes) are treated as commits
fn parse_version(s: &str) -> DuckDBVersion {
    let s = s.trim();

    // Strip leading 'v' if present
    let version_str = s.strip_prefix('v').unwrap_or(s);

    // Check if it looks like a semver release (X.Y.Z pattern)
    let parts: Vec<&str> = version_str.split('.').collect();
    if parts.len() >= 2 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())) {
        DuckDBVersion::Release(version_str.to_owned())
    } else {
        panic!("Invalid DuckDB version: {s}");
    }
}

/// Create an HTTP client with appropriate timeout settings.
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

/// Download a URL with retry logic for transient failures (5xx, timeouts, connection errors).
fn download_with_retries(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let client = http_client()?;
    let max_retries = 3;

    for attempt in 1..=max_retries {
        match client.get(url).send() {
            Ok(response) if response.status().is_success() => {
                return Ok(response.bytes()?.to_vec());
            }
            Ok(response) if response.status().is_server_error() => {
                println!(
                    "cargo:warning=Download attempt {attempt}/{max_retries} failed: HTTP {} for {url}",
                    response.status()
                );
            }
            Ok(response) => {
                // Client errors (4xx) are not retryable
                return Err(format!("Failed to download {url}: HTTP {}", response.status()).into());
            }
            Err(e) => {
                println!(
                    "cargo:warning=Download attempt {attempt}/{max_retries} failed: {e} for {url}"
                );
            }
        }

        if attempt < max_retries {
            let delay = std::time::Duration::from_secs(2u64.pow(attempt as u32));
            println!("cargo:warning=Retrying in {}s...", delay.as_secs());
            std::thread::sleep(delay);
        }
    }

    Err(format!("Failed to download {url} after {max_retries} attempts").into())
}

/// Get the target-specific platform and architecture for downloading prebuilt libraries.
fn platform_arch() -> Result<(&'static str, &'static str), Box<dyn std::error::Error>> {
    let target = env::var("TARGET")?;
    match target.as_str() {
        "aarch64-apple-darwin" | "x86_64-apple-darwin" => Ok(("osx", "universal")),
        "x86_64-unknown-linux-gnu" => Ok(("linux", "amd64")),
        "aarch64-unknown-linux-gnu" => Ok(("linux", "arm64")),
        _ => Err(format!("Unsupported target: {target}").into()),
    }
}

/// Get the base target directory for DuckDB artifacts.
fn target_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    Ok(manifest_dir.parent().unwrap().join("target"))
}

/// Download prebuilt DuckDB libraries from GitHub releases.
/// Only valid for release versions.
fn download_duckdb_lib_archive() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let version = match &*DUCKDB_VERSION {
        DuckDBVersion::Release(v) => v,
        DuckDBVersion::Commit(_) => {
            return Err("Cannot download prebuilt libraries for commit hashes".into());
        }
    };

    let target_dir = target_dir()?;
    let duckdb_lib_dir = target_dir.join(format!("duckdb-lib-v{version}"));

    let (platform, arch) = platform_arch()?;
    let archive_name = format!("libduckdb-{platform}-{arch}.zip");
    let url = format!("{DUCKDB_RELEASES_URL}/v{version}/{archive_name}");
    let archive_path = duckdb_lib_dir.join(&archive_name);

    // Create directory if it doesn't exist
    fs::create_dir_all(&duckdb_lib_dir)?;

    // Download if archive doesn't exist
    if !archive_path.exists() {
        println!("cargo:info=Downloading DuckDB libraries from {url}");
        let bytes = download_with_retries(&url)?;
        fs::write(&archive_path, &bytes)?;
        println!("cargo:info=Downloaded to {}", archive_path.display());
    }

    Ok(archive_path)
}

/// Extract DuckDB libraries from the downloaded archive.
fn extract_duckdb_libraries(archive_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let duckdb_lib_dir = archive_path
        .parent()
        .ok_or("Invalid archive path")?
        .to_path_buf();

    // Check if already extracted (check for both .dylib and .so)
    let dylib_exists = duckdb_lib_dir.join("libduckdb.dylib").exists();
    let so_exists = duckdb_lib_dir.join("libduckdb.so").exists();

    if dylib_exists || so_exists {
        println!("cargo:info=DuckDB libraries already extracted, skipping");
        return Ok(duckdb_lib_dir);
    }

    println!(
        "cargo:info=Extracting DuckDB libraries to {}",
        duckdb_lib_dir.display()
    );
    let file = fs::File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    archive.extract(&duckdb_lib_dir)?;

    Ok(duckdb_lib_dir)
}

/// Download DuckDB source code archive from GitHub.
/// Works for both release versions and commit hashes.
fn download_duckdb_source_archive() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let target_dir = target_dir()?;
    let source_dir = target_dir.join(format!("duckdb-source-{}", DUCKDB_VERSION.dir_suffix()));

    let url = match &*DUCKDB_VERSION {
        DuckDBVersion::Release(v) => {
            format!("https://github.com/duckdb/duckdb/archive/refs/tags/v{v}.zip")
        }
        DuckDBVersion::Commit(c) => {
            format!("https://github.com/duckdb/duckdb/archive/{c}.zip")
        }
    };

    let archive_path = source_dir.with_extension("zip");

    // Create directory if it doesn't exist
    fs::create_dir_all(&source_dir)?;

    // Download if archive doesn't exist
    if !archive_path.exists() {
        println!("cargo:info=Downloading DuckDB source code from {url}");
        let bytes = download_with_retries(&url)?;
        fs::write(&archive_path, &bytes)?;
        println!("cargo:info=Downloaded to {}", archive_path.display());
    }

    Ok(source_dir)
}

/// Extract DuckDB source code from the downloaded archive.
fn extract_duckdb_source(source_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let archive_path = source_dir.with_extension("zip");

    // Check if already extracted by looking for CMakeLists.txt
    let inner_dir_name = DUCKDB_VERSION.archive_inner_dir_name();
    let cmake_file = source_dir.join(&inner_dir_name).join("CMakeLists.txt");

    if cmake_file.exists() {
        println!("cargo:info=DuckDB source already extracted, skipping");
        return Ok(source_dir.to_path_buf());
    }

    println!(
        "cargo:info=Extracting DuckDB source to {}",
        source_dir.display()
    );
    let file = fs::File::open(&archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Extract all files, the archive contains a root directory like `duckdb-{version}/`
    archive.extract(source_dir)?;

    Ok(source_dir.to_path_buf())
}

/// Build DuckDB from source. Used for commit hashes or when VX_DUCKDB_DEBUG is set.
fn build_duckdb(
    duckdb_source_dir: &Path,
    version: &DuckDBVersion,
    debug: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let build_type = match debug {
        true => "debug",
        false => "release",
    };
    // Check for ninja
    if Command::new("ninja").arg("--version").output().is_err() {
        return Err(
            "'ninja' is required to build DuckDB. Install it via your package manager.".into(),
        );
    }

    let inner_dir_name = DUCKDB_VERSION.archive_inner_dir_name();
    let duckdb_repo_dir = duckdb_source_dir.join(&inner_dir_name);
    let build_dir = duckdb_repo_dir.join("build").join(build_type);

    let lib_dir = build_dir.join("src");
    let lib_dir_str = lib_dir.display();
    println!("cargo:info=Checking if DuckDB is already built in {lib_dir_str}",);

    let already_built = lib_dir.join("libduckdb.dylib").exists()
        || lib_dir.join("libduckdb.so").exists()
        || lib_dir
            .read_dir()
            .map(|mut d| {
                d.any(|e| e.is_ok_and(|e| e.file_name().to_string_lossy().starts_with("libduckdb")))
            })
            .unwrap_or(false);

    if !already_built {
        println!("cargo:info=Building DuckDB from source (this may take a while)...");

        // Build with ASAN/TSAN if VX_DUCKDB_SAN=1
        let (asan_option, tsan_option) =
            if env::var("VX_DUCKDB_SAN").is_ok_and(|v| matches!(v.as_str(), "1" | "true")) {
                ("0", "1") // DISABLE_SANITIZER=0 enables ASAN, THREADSAN=1 enables TSAN
            } else {
                ("1", "0")
            };

        let mut envs = vec![
            ("GEN", "ninja"),
            ("DISABLE_SANITIZER", asan_option),
            ("THREADSAN", tsan_option),
            ("BUILD_SHELL", "false"),
            ("BUILD_UNITTESTS", "false"),
            ("ENABLE_UNITTEST_CPP_TESTS", "false"),
        ];

        // If we're building from a commit (likely a pre-release), we need to
        // build extensions statically. Otherwise DuckDB tries to load them
        // from an http endpoint with version 0.0.1 (all non-tagged builds)
        // which doesn't exists. httpfs also requires CURL dev headers
        if matches!(version, DuckDBVersion::Commit(_)) {
            envs.push(("BUILD_EXTENSIONS", "httpfs;parquet;tpch;tpcds;jemalloc"));
        };

        let output = Command::new("make")
            .current_dir(&duckdb_repo_dir)
            .envs(envs)
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Failed to build DuckDB:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        println!("cargo:info=DuckDB build completed successfully");
    } else {
        println!("cargo:info=DuckDB already built, skipping build");
    }

    // Copy libraries to a stable location
    let target_dir = target_dir()?;
    let duckdb_library_dir = target_dir.join(format!("duckdb-lib-{}", DUCKDB_VERSION.dir_suffix()));

    // Only copy if the destination doesn't have the libraries
    if !(duckdb_library_dir.join("libduckdb.dylib").exists()
        || duckdb_library_dir.join("libduckdb.so").exists())
    {
        // Clean and recreate destination
        match fs::remove_dir_all(&duckdb_library_dir) {
            Err(err) if err.kind() == ErrorKind::NotFound => (),
            otherwise => otherwise?,
        }
        fs::create_dir_all(&duckdb_library_dir)?;

        // Copy .dylib and .so files
        for entry in fs::read_dir(lib_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libduckdb"))
            {
                let dest = duckdb_library_dir.join(entry.file_name());
                fs::copy(&path, &dest)?;
            }
        }
    }

    Ok(duckdb_library_dir)
}

/// Get the path to the DuckDB include directory within the source tree.
fn duckdb_include_path(source_dir: &Path) -> PathBuf {
    let inner_dir_name = DUCKDB_VERSION.archive_inner_dir_name();
    source_dir.join(inner_dir_name).join("src").join("include")
}

fn main() {
    // Emit rerun-if-env-changed for all relevant environment variables
    println!("cargo:rerun-if-env-changed=DUCKDB_VERSION");
    println!("cargo:rerun-if-env-changed=VX_DUCKDB_DEBUG");
    println!("cargo:rerun-if-env-changed=VX_DUCKDB_SAN");
    println!("cargo:rerun-if-env-changed=CARGO_HTTP_TIMEOUT");
    println!("cargo:rerun-if-env-changed=HTTP_TIMEOUT");
    println!("cargo:rerun-if-env-changed=TARGET");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let duckdb_symlink = crate_dir.join("duckdb");

    match &*DUCKDB_VERSION {
        DuckDBVersion::Release(v) => println!("cargo:info=Using DuckDB release version: {v}"),
        DuckDBVersion::Commit(c) => println!("cargo:info=Using DuckDB commit: {c}"),
    }

    // Always download and extract source (needed for headers)
    let source_dir = download_duckdb_source_archive().unwrap();
    let extracted_source_path = extract_duckdb_source(&source_dir).unwrap();

    // Create/update symlink to source directory
    // Remove existing symlink/directory first (ignore errors if they don't exist)
    drop(fs::remove_file(&duckdb_symlink));
    drop(fs::remove_dir_all(&duckdb_symlink));
    std::os::unix::fs::symlink(&extracted_source_path, &duckdb_symlink).unwrap();

    let use_debug_build =
        env::var("VX_DUCKDB_DEBUG").is_ok_and(|v| matches!(v.as_str(), "1" | "true"));
    println!("cargo:info=DuckDB debug build: {use_debug_build}");

    let library_path = if use_debug_build || !DUCKDB_VERSION.is_release() {
        // Build from source for:
        // - Commit hashes (no prebuilt available)
        // - When VX_DUCKDB_DEBUG=1 (user wants debug build)
        match build_duckdb(&extracted_source_path, &DUCKDB_VERSION, use_debug_build) {
            Ok(path) => path,
            Err(err) => {
                println!("cargo:error={err}");
                panic!("duckdb build failed");
            }
        }
    } else {
        // Download prebuilt libraries for release versions
        let archive_path = download_duckdb_lib_archive().unwrap();
        extract_duckdb_libraries(&archive_path).unwrap()
    };

    let duckdb_include_path = duckdb_include_path(&extracted_source_path);

    // Generate the _imported_ bindings from our C++ code.
    bindgen::Builder::default()
        .header("cpp/include/duckdb_vx.h")
        .override_abi(Abi::CUnwind, ".*")
        // Allow for auto-generated cpp.rs code.
        .raw_line("#![allow(dead_code)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(clippy::suspicious_doc_comments)]")
        .raw_line("#![allow(clippy::enum_variant_names)]")
        // Add the #[must_use] attribute to FFI functions that return results.
        .must_use_type("duckdb_state")
        .rustified_enum("duckdb_state")
        .rustified_enum("DUCKDB_VX_EXPR_CLASS")
        .rustified_enum("DUCKDB_VX_EXPR_TYPE")
        .rustified_enum("DUCKDB_VX_TABLE_FILTER_TYPE")
        .rustified_enum("DUCKDB_VX_VECTOR_TYPE")
        .rustified_non_exhaustive_enum("DUCKDB_TYPE")
        .size_t_is_usize(true)
        .clang_arg(format!("-I{}", duckdb_include_path.display()))
        .clang_arg(format!("-I{}", crate_dir.join("cpp/include").display()))
        .generate_comments(true)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(crate_dir.join("src/cpp.rs"))
        .expect("Couldn't write bindings!");

    // Link against DuckDB dylib.
    println!("cargo:rustc-link-search=native={}", library_path.display());
    println!("cargo:rustc-link-lib=dylib=duckdb");

    // Set rpath for binaries built directly from this crate (this is not inherited by downstream crates).
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_path.display());

    // Export the library path for downstream crates via the `links` manifest key.
    // Downstream crates can access this via `env::var("DEP_DUCKDB_LIB_DIR")` in their build.rs
    // and add their own rpath:
    //
    //   if let Ok(duckdb_lib) = env::var("DEP_DUCKDB_LIB_DIR") {
    //       println!("cargo:rustc-link-arg=-Wl,-rpath,{duckdb_lib}");
    //   }
    //
    // Alternatively, set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS) at runtime:
    //   LD_LIBRARY_PATH=/path/to/target/duckdb-lib-vX.Y.Z cargo run --bin ...
    //
    println!("cargo:lib_dir={}", library_path.display());

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
        .cpp(true)
        .include(&duckdb_include_path)
        .include("cpp/include")
        .file("cpp/client_context.cpp")
        .file("cpp/config.cpp")
        .file("cpp/copy_function.cpp")
        .file("cpp/data.cpp")
        .file("cpp/data_chunk.cpp")
        .file("cpp/error.cpp")
        .file("cpp/expr.cpp")
        .file("cpp/file_system.cpp")
        .file("cpp/logical_type.cpp")
        .file("cpp/object_cache.cpp")
        .file("cpp/reusable_dict.cpp")
        .file("cpp/replacement_scan.cpp")
        .file("cpp/scalar_function.cpp")
        .file("cpp/table_filter.cpp")
        .file("cpp/table_function.cpp")
        .file("cpp/value.cpp")
        .file("cpp/vector.cpp")
        .file("cpp/vector_buffer.cpp")
        .compile("vortex-duckdb-extras");

    // Generate the _exported_ bindings from our Rust code.
    let generated_header = crate_dir.join("include/vortex.h");
    cbindgen::Builder::new()
        .with_config(cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).unwrap())
        .with_crate(&crate_dir)
        .with_no_includes()
        .generate()
        .expect("error: Unable to generate bindings for vortex.h")
        .write_to_file(&generated_header);

    // Run clang-format on the generated header.
    if let Ok(status) = Command::new("clang-format")
        .arg("-i")
        .arg("--style=file")
        .arg(&generated_header)
        .status()
    {
        if !status.success() {
            println!("cargo:warning=clang-format exited with status {status}");
        }
    } else {
        println!("cargo:warning=clang-format not found, skipping formatting of generated header");
    }

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
