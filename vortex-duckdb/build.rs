// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
// exit(1) + cargo:error= doesn't provide a double-traceback like panic!()
#![expect(clippy::exit)]

use std::env;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::exit;

use bindgen::Abi;

const DUCKDB_RELEASES_URL: &str = "https://github.com/duckdb/duckdb/releases/download";
const DUCKDB_SOURCE_RELEASE_URL: &str = "https://github.com/duckdb/duckdb/archive/refs/tags";
const DUCKDB_SOURCE_COMMIT_URL: &str = "https://github.com/duckdb/duckdb/archive";

const BUILD_ARTIFACTS: [&str; 3] = ["libduckdb.dylib", "libduckdb.so", "libduckdb_static.a"];

const SOURCE_FILES: [&str; 17] = [
    "cpp/client_context.cpp",
    "cpp/config.cpp",
    "cpp/copy_function.cpp",
    "cpp/data.cpp",
    "cpp/data_chunk.cpp",
    "cpp/error.cpp",
    "cpp/expr.cpp",
    "cpp/file_system.cpp",
    "cpp/logical_type.cpp",
    "cpp/replacement_scan.cpp",
    "cpp/reusable_dict.cpp",
    "cpp/scalar_function.cpp",
    "cpp/table_filter.cpp",
    "cpp/table_function.cpp",
    "cpp/value.cpp",
    "cpp/vector.cpp",
    "cpp/vector_buffer.cpp",
];

const DOWNLOAD_MAX_RETRIES: i32 = 3;
const DOWNLOAD_TIMEOUT: u64 = 90;

#[derive(Debug, Clone)]
enum DuckDBVersion {
    Release(String), // i.e. X.Y.Z. Download pre-compiled libraries from GitHub releases.
    Commit(String),  // Download source and build DuckDB from scratch.
}

impl std::fmt::Display for DuckDBVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuckDBVersion::Release(v) => write!(f, "v{v}"),
            DuckDBVersion::Commit(c) => write!(f, "{c}"),
        }
    }
}

impl DuckDBVersion {
    /// Returns the name of the extracted source directory inside the zip archive.
    /// GitHub archives extract to `duckdb-{version}` for tags and `duckdb-{commit}` for commits.
    fn archive_inner_dir_name(&self) -> String {
        match self {
            DuckDBVersion::Release(v) => format!("duckdb-{v}"),
            DuckDBVersion::Commit(c) => format!("duckdb-{c}"),
        }
    }
}

impl From<&String> for DuckDBVersion {
    fn from(s: &String) -> Self {
        let s = s.trim();
        let version_str = s.strip_prefix('v').unwrap_or(s);
        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.len() >= 2 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())) {
            DuckDBVersion::Release(version_str.to_owned())
        } else {
            DuckDBVersion::Commit(version_str.to_owned())
        }
    }
}

fn download_url(url: &str, path: &Path) {
    if path.exists() {
        return;
    }
    println!("cargo:info=Downloading DuckDB from {url}");

    let timeout_secs = env::var("CARGO_HTTP_TIMEOUT")
        .or_else(|_| env::var("HTTP_TIMEOUT"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DOWNLOAD_TIMEOUT);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .unwrap();

    for attempt in 1..=DOWNLOAD_MAX_RETRIES {
        match client.get(url).send() {
            Ok(response) if response.status().is_success() => {
                let bytes = response.bytes().unwrap().to_vec();
                fs::write(path, bytes).unwrap();
                println!("cargo:info=Downloaded to {url}");
                return;
            }
            Ok(response) if response.status().is_server_error() => {
                let status = response.status();
                println!(
                    "cargo:warning=Download attempt \
                    {attempt}/{DOWNLOAD_MAX_RETRIES} failed: HTTP {status} for {url}"
                );
            }
            Err(e) => {
                println!(
                    "cargo:warning=Download attempt \
                    {attempt}/{DOWNLOAD_MAX_RETRIES} failed: {e}"
                );
            }
            // Client errors (4xx) are not retryable
            Ok(response) => {
                let status = response.status();
                println!("cargo:error=Failed to download {url}: HTTP {status}");
                exit(1);
            }
        }

        if attempt < DOWNLOAD_MAX_RETRIES {
            let delay = std::time::Duration::from_secs(2u64.pow(attempt as u32));
            println!("cargo:warning=Retrying in {}s...", delay.as_secs());
            std::thread::sleep(delay);
        }
    }

    println!("cargo:error=Failed to download {url} after {DOWNLOAD_MAX_RETRIES} attempts");
    exit(1);
}

fn extract(archive: &Path, dest: &Path) {
    println!(
        "cargo:info=Extracting {} to {}",
        archive.display(),
        dest.display()
    );
    let file = fs::File::open(archive).unwrap();
    zip::ZipArchive::new(file).unwrap().extract(dest).unwrap();
}

fn download(version: &DuckDBVersion, library_dir: &Path) {
    let target = env::var("TARGET").unwrap();
    let (platform, arch) = match target.as_str() {
        "aarch64-apple-darwin" | "x86_64-apple-darwin" => ("osx", "universal"),
        "x86_64-unknown-linux-gnu" => ("linux", "amd64"),
        "aarch64-unknown-linux-gnu" => ("linux", "arm64"),
        _ => {
            println!("cargo:error=Unsupported target {target}");
            exit(1);
        }
    };

    let archive_name = format!("libduckdb-{platform}-{arch}.zip");
    let url = format!("{DUCKDB_RELEASES_URL}/{version}/{archive_name}");
    let archive_path = library_dir.join(&archive_name);

    fs::create_dir_all(library_dir).unwrap();
    download_url(&url, &archive_path);

    let duckdb_lib_dir = archive_path.parent().unwrap().to_path_buf();
    for artifact in BUILD_ARTIFACTS {
        if duckdb_lib_dir.join(artifact).exists() {
            return;
        }
    }
    extract(&archive_path, &duckdb_lib_dir);
}

fn build_duckdb(version: &DuckDBVersion, duckdb_repo_dir: &Path) {
    if let Err(e) = Command::new("make").arg("--version").output() {
        println!("cargo:error=make is required to build DuckDB: {e}");
        exit(1);
    }
    if let Err(e) = Command::new("ninja").arg("--version").output() {
        println!("cargo:error=ninja is required to build DuckDB: {e}");
        exit(1);
    }

    println!("cargo:info=Building DuckDB from source (this may take a while)...");
    let (asan_option, tsan_option) =
        if env::var("VX_DUCKDB_SAN").is_ok_and(|v| matches!(v.as_str(), "1" | "true")) {
            ("0", "1") // DISABLE_SANITIZER=0 enables ASAN, THREADSAN=1 enables TSAN
        } else {
            ("1", "0")
        };

    // If we're building from a commit  we need to build httpfs and benchmark
    // extensions statically, otherwise DuckDB tries to load them from an http
    // endpoint with version 0.0.1 (all non-tagged builds) which doesn't exist.
    // httpfs static build also requires CURL dev headers
    let static_extensions = match version {
        DuckDBVersion::Release(_) => "parquet;jemalloc",
        DuckDBVersion::Commit(_) => "parquet;jemalloc;httpfs;tpch;tpcds",
    };

    let envs = [
        ("GEN", "ninja"),
        ("DISABLE_SANITIZER", asan_option),
        ("THREADSAN", tsan_option),
        ("BUILD_SHELL", "false"),
        ("BUILD_UNITTESTS", "false"),
        ("ENABLE_UNITTEST_CPP_TESTS", "false"),
        ("BUILD_EXTENSIONS", static_extensions),
    ];

    let output = Command::new("make")
        .current_dir(duckdb_repo_dir)
        .envs(envs)
        .output()
        .unwrap();
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("cargo:error=Failed to build DuckDB:\nstdout: {stdout}\nstderr: {stderr}");
        exit(1);
    }

    println!("cargo:info=DuckDB build completed");
}

fn try_build_duckdb(
    source_dir: &Path,
    library_dir: &Path,
    version: &DuckDBVersion,
    build_type: &str,
) {
    let inner_dir_name = version.archive_inner_dir_name();
    let repo_dir = source_dir.join(&inner_dir_name);
    let build_dir = repo_dir.join("build").join(build_type);
    let build_src_dir = build_dir.join("src");

    let mut build = true;
    for artifact in BUILD_ARTIFACTS {
        let path = build_src_dir.join(artifact);
        if path.exists() {
            println!("cargo:info=Found {artifact} in {}", path.display());
            build = false;
            break;
        }
    }

    if build {
        build_duckdb(version, &repo_dir);
    }

    let library_dir_str = library_dir.display();
    if let Err(err) = fs::remove_dir_all(library_dir)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        println!("cargo:error=Failed to remove {library_dir_str}: {err}");
        exit(1);
    };
    fs::create_dir_all(library_dir).unwrap();

    let mut found_artifact = false;
    for artifact in BUILD_ARTIFACTS {
        let src = build_src_dir.join(artifact);
        if !src.exists() {
            continue;
        }
        let dest = library_dir.join(artifact);
        fs::copy(&src, &dest).unwrap();
        found_artifact = true;
    }

    if !found_artifact {
        let artifacts = BUILD_ARTIFACTS.join(",");
        println!("cargo:error=Failed to find any of {artifacts} after build");
        exit(1);
    }
}

fn c2rust(crate_dir: &Path, duckdb_include_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header("cpp/include/duckdb_vx.h")
        .override_abi(Abi::CUnwind, ".*")
        // Allow for auto-generated cpp.rs code.
        .raw_line("#![allow(dead_code)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(clippy::absolute_paths)]")
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
        .clang_arg(format!("-I{}", duckdb_include_dir.display()))
        .clang_arg(format!("-I{}", crate_dir.join("cpp/include").display()))
        .generate_comments(true)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate();

    let bindings = match bindings {
        Ok(b) => b,
        Err(e) => {
            println!("cargo:error=Failed to generate Rust bindings: {e}");
            exit(1);
        }
    };
    let out_path = crate_dir.join("src/cpp.rs");
    let new_contents = bindings.to_string();
    let write = match fs::read_to_string(&out_path) {
        Ok(existing) => existing != new_contents,
        Err(_) => true,
    };
    if write && let Err(e) = fs::write(&out_path, new_contents) {
        println!("cargo:error=Failed to write Rust bindings: {e}");
        exit(1);
    }
}

fn cpp(duckdb_include_dir: &Path) {
    cc::Build::new()
        .std("c++20")
        .flags(["-Wall", "-Wextra", "-Wpedantic"])
        .cpp(true)
        .include(duckdb_include_dir)
        .include("cpp/include")
        .files(SOURCE_FILES)
        .compile("vortex-duckdb-extras");
    // bindgen generates rerun-if-changed for .h/.hpp files
    for e in SOURCE_FILES {
        println!("cargo:rerun-if-changed={e}");
    }
}

fn rust2c(crate_dir: &Path) {
    let header = crate_dir.join("include/vortex.h");
    let output = cbindgen::Builder::new()
        .with_config(cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).unwrap())
        .with_crate(crate_dir)
        .with_no_includes()
        .generate();
    match output {
        Ok(bindings) => bindings.write_to_file(&header),
        Err(e) => {
            println!("cargo:error=Failed to generate cbindgen bindings for vortex.h: {e}");
            exit(1);
        }
    };

    let mut cmd = Command::new("clang-format");
    let format = cmd.arg("-i").arg("--style=file").arg(&header);
    if let Ok(status) = format.status() {
        if !status.success() {
            println!("cargo:warning=clang-format exited with status {status}");
        }
    } else {
        println!("cargo:warning=clang-format not found, skipping formatting of generated header");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=DUCKDB_VERSION");
    println!("cargo:rerun-if-env-changed=VX_DUCKDB_DEBUG");
    println!("cargo:rerun-if-env-changed=VX_DUCKDB_SAN");
    println!("cargo:rerun-if-env-changed=CARGO_HTTP_TIMEOUT");
    println!("cargo:rerun-if-env-changed=HTTP_TIMEOUT");
    println!("cargo:rerun-if-env-changed=TARGET");

    // `DUCKDB_VERSION` is set by the extension build in CI.
    // This is to ensure we don't implicitly build against a different DuckDB
    // version during an extension build which might lead to subtle ABI breaks,
    // e.g. reordering fields in C++ structs.
    let version = env::var("DUCKDB_VERSION")
        // You can also change this version to a commit hash
        .unwrap_or_else(|_| "1.5.2".to_owned());
    let version = DuckDBVersion::from(&version);
    match &version {
        DuckDBVersion::Release(v) => println!("cargo:info=Using DuckDB release version: {v}"),
        DuckDBVersion::Commit(c) => println!("cargo:info=Using DuckDB commit: {c}"),
    }

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let duckdb_dir = crate_dir.join("duckdb");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let library_dir = out_dir.join(format!("duckdb-lib-{version}"));

    let library_dir_str = library_dir.display();
    println!("cargo:rustc-link-search=native={library_dir_str}");
    println!("cargo:rustc-link-lib=dylib=duckdb");

    // Set rpath for binaries built directly from this crate. This is not
    // inherited by downstream crates.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{library_dir_str}");

    // Export the library path for downstream crates via the `links` manifest key.
    // Downstream crates can access this via `env::var("DEP_DUCKDB_LIB_DIR")` in their build.rs
    // and add their own rpath:
    //
    //   if let Ok(duckdb_lib) = env::var("DEP_DUCKDB_LIB_DIR") {
    //       println!("cargo:rustc-link-arg=-Wl,-rpath,{duckdb_lib}");
    //   }
    //
    // Alternatively, set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS) at runtime.
    println!("cargo:lib_dir={library_dir_str}");

    let source_dir = out_dir.join(format!("duckdb-source-{version}"));
    let source_archive_url = match &version {
        DuckDBVersion::Release(v) => format!("{DUCKDB_SOURCE_RELEASE_URL}/v{v}.zip"),
        DuckDBVersion::Commit(c) => format!("{DUCKDB_SOURCE_COMMIT_URL}/{c}.zip"),
    };

    fs::create_dir_all(&source_dir).unwrap();
    let source_archive_path = source_dir.with_extension("zip");
    download_url(&source_archive_url, &source_archive_path);

    let inner_dir = source_dir.join(version.archive_inner_dir_name());
    if !inner_dir.join("CMakeLists.txt").exists() {
        extract(&source_archive_path, &source_dir);
    }

    drop(fs::remove_file(&duckdb_dir));
    drop(fs::remove_dir_all(&duckdb_dir));
    symlink(&source_dir, &duckdb_dir).unwrap();

    let has_debug_env =
        env::var("VX_DUCKDB_DEBUG").is_ok_and(|v| matches!(v.as_str(), "1" | "true"));
    let build_type = match has_debug_env {
        true => "debug",
        false => "release",
    };
    println!("cargo:info=building DuckDB in {build_type} mode");

    if has_debug_env || matches!(version, DuckDBVersion::Commit(_)) {
        try_build_duckdb(&source_dir, &library_dir, &version, build_type);
    } else {
        download(&version, &library_dir);
    };

    let duckdb_include_dir = inner_dir.join("src").join("include");
    c2rust(&crate_dir, &duckdb_include_dir);
    cpp(&duckdb_include_dir);
    rust2c(&crate_dir);
}
