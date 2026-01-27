// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Downloads the nvCOMP SDK and generates Rust FFI bindings for nvCOMP Zstd.
//!
//! Bindings are generated unconditionally. This allows for development against the
//! CUDA APIs in environments that don't support CUDA.
//!
//! The library is loaded at runtime via libloading.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::panic)]

use std::env;
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

use xz2::read::XzDecoder;

const NVCOMP_VERSION: &str = "5.1.0.21";
const CUDA_VERSION: &str = "cuda12";

/// Minimal CUDA runtime stub header for bindgen.
const CUDA_RUNTIME_STUB: &str = r#"
#pragma once
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
"#;

/// Minimal nvCOMP headers for non-Linux platforms to allow bindgen to run.
const NVCOMP_STUB_HEADER: &str = r#"
#pragma once
#include <stddef.h>
#include "cuda_runtime.h"

typedef enum nvcompStatus_t {
    nvcompSuccess = 0,
    nvcompErrorInvalidValue = 1,
    nvcompErrorNotSupported = 2,
    nvcompErrorCannotDecompress = 3,
    nvcompErrorBadChecksum = 4,
    nvcompErrorCannotVerifyChecksums = 5,
    nvcompErrorOutputBufferTooSmall = 6,
    nvcompErrorWrongHeaderLength = 7,
    nvcompErrorAlignment = 8,
    nvcompErrorChunkSizeTooLarge = 9,
    nvcompErrorCannotCompress = 10,
    nvcompErrorWrongInputLength = 11,
    nvcompErrorBatchSizeTooLarge = 12,
    nvcompErrorCudaError = 13,
    nvcompErrorInternal = 14
} nvcompStatus_t;

typedef enum nvcompDecompressBackend_t {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT = 0,
    NVCOMP_DECOMPRESS_BACKEND_HARDWARE = 1,
    NVCOMP_DECOMPRESS_BACKEND_CUDA = 2
} nvcompDecompressBackend_t;

typedef struct nvcompBatchedZstdDecompressOpts_t {
    nvcompDecompressBackend_t backend;
    unsigned char reserved[60];
} nvcompBatchedZstdDecompressOpts_t;
"#;

const NVCOMP_ZSTD_STUB_HEADER: &str = r#"
#pragma once
#include "nvcomp.h"

nvcompStatus_t nvcompBatchedZstdDecompressGetTempSizeAsync(
    size_t numChunks,
    size_t maxUncompressedChunkBytes,
    nvcompBatchedZstdDecompressOpts_t opts,
    size_t* tempBytes,
    size_t maxTotalUncompressedBytes);

nvcompStatus_t nvcompBatchedZstdDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t num_chunks,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompBatchedZstdDecompressOpts_t opts,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);
"#;

fn main() {
    // Declare the cfg so rustc doesn't warn about unexpected cfg.
    println!("cargo::rustc-check-cfg=cfg(cuda_available)");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcomp_dir = out_dir.join("nvcomp-sdk");

    // Create CUDA stub header in OUT_DIR for bindgen
    let cuda_stub_dir = out_dir.join("cuda-stub");
    fs::create_dir_all(&cuda_stub_dir).unwrap();
    fs::write(cuda_stub_dir.join("cuda_runtime.h"), CUDA_RUNTIME_STUB).unwrap();

    let is_linux = env::consts::OS == "linux";
    let include_dir = if is_linux {
        let (os, arch) = match (env::consts::OS, env::consts::ARCH) {
            ("linux", "x86_64") => ("linux", "x86_64"),
            ("linux", "aarch64") => ("linux", "sbsa"),
            _ => ("linux", "x86_64"),
        };

        let archive_name = format!("nvcomp-{os}-{arch}-{NVCOMP_VERSION}_{CUDA_VERSION}-archive");
        let url = format!(
            "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/{os}-{arch}/{archive_name}.tar.xz"
        );

        let include_dir = nvcomp_dir.join("include");

        if !include_dir.exists() {
            let response = reqwest::blocking::get(&url)
                .unwrap_or_else(|e| panic!("Failed to download nvCOMP: {e}"));

            assert!(
                response.status().is_success(),
                "Failed to download nvCOMP: HTTP {}",
                response.status()
            );

            let bytes = response.bytes().unwrap();

            // Extract tar.xz archive.
            let cursor = Cursor::new(bytes.as_ref());
            let xz = XzDecoder::new(cursor);
            let mut archive = tar::Archive::new(xz);

            let temp_dir = nvcomp_dir.with_extension("tmp");
            fs::create_dir_all(&temp_dir).unwrap();
            archive.unpack(&temp_dir).unwrap();

            // Move extracted content.
            let extracted = temp_dir.join(&archive_name);
            if nvcomp_dir.exists() {
                fs::remove_dir_all(&nvcomp_dir).unwrap();
            }
            fs::rename(&extracted, &nvcomp_dir).unwrap();
            fs::remove_dir_all(&temp_dir).ok();
        }

        include_dir
    } else {
        let stub_include = out_dir.join("nvcomp-stub").join("include");
        let stub_nvcomp = stub_include.join("nvcomp");
        fs::create_dir_all(&stub_nvcomp).unwrap();
        fs::write(stub_include.join("nvcomp.h"), NVCOMP_STUB_HEADER).unwrap();
        fs::write(stub_nvcomp.join("zstd.h"), NVCOMP_ZSTD_STUB_HEADER).unwrap();
        stub_include
    };

    // Functions are loaded at runtime via libloading to avoid link-time symbol resolution.
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("nvcomp.h").to_string_lossy())
        .header(include_dir.join("nvcomp/zstd.h").to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", cuda_stub_dir.display()))
        .allowlist_type("nvcompStatus_t")
        .allowlist_type("nvcompBatchedZstdDecompressOpts_t")
        .allowlist_type("nvcompDecompressBackend_t")
        .allowlist_function("nvcompBatchedZstdDecompressGetTempSizeAsync")
        .allowlist_function("nvcompBatchedZstdDecompressAsync")
        .dynamic_library_name("NvcompLibrary")
        .dynamic_link_require_all(true)
        .wrap_unsafe_ops(true)
        .blocklist_type("CUstream_st")
        .blocklist_type("cudaStream_t")
        .raw_line("// FFI type definitions for nvCOMP (generated by bindgen).")
        .raw_line("// Functions are loaded at runtime via libloading.")
        .raw_line("")
        .raw_line("pub type cudaStream_t = *mut std::ffi::c_void;")
        .generate()
        .expect("Failed to generate nvcomp bindings");

    bindings.write_to_file(out_dir.join("sys.rs")).unwrap();

    // Set cuda_available cfg if CUDA is detected on the system.
    // This gates tests and benchmarks that require CUDA at runtime.
    if cuda_available() {
        println!("cargo:rustc-cfg=cuda_available");
    }
}

/// Check if CUDA is available based on nvcc.
fn cuda_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}
