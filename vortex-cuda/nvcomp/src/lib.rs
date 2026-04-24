// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust bindings to NVIDIA nvCOMP compression library.
//!
//! This crate provides bindings to nvCOMP, with the library loaded at runtime
//! via `libloading`. This allows the crate to compile on systems without CUDA
//! or nvcomp installed - the library is only required at runtime when the
//! functions are actually called.
//!
//! # Platform Support
//!
//! nvCOMP is only available on Linux x86_64 and ARM64. On other platforms,
//! this crate still compiles but will fail at runtime when trying to load
//! the library.
//!
//! # Runtime Requirements
//!
//! The nvcomp library must be available at runtime.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Raw FFI type definitions and dynamically-loaded function pointers from bindgen.
#[expect(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    clippy::all
)]
pub mod sys;

mod error;
pub mod zstd;

pub use error::NvcompError;

/// The loaded nvcomp library instance.
static NVCOMP_LIB: OnceLock<Result<sys::NvcompLibrary, String>> = OnceLock::new();

fn load_nvcomp() -> Result<sys::NvcompLibrary, String> {
    let lib_name = "libnvcomp.so";
    let build_lib_dir = env!("OUT_DIR");
    let sdk_lib_path = PathBuf::from(build_lib_dir)
        .join("nvcomp-sdk")
        .join("lib")
        .join(lib_name);

    // SAFETY: The library at the SDK path is a valid nvcomp shared library
    // downloaded during the build process.
    unsafe {
        sys::NvcompLibrary::new(&sdk_lib_path)
            .map_err(|e| format!("Failed to load nvcomp library: {e}"))
    }
}

/// Gets a reference to the loaded nvcomp library.
///
/// The library is loaded lazily on first access. Returns an error if the
/// library cannot be found or loaded.
pub fn nvcomp_library() -> Result<&'static sys::NvcompLibrary, NvcompError> {
    NVCOMP_LIB
        .get_or_init(load_nvcomp)
        .as_ref()
        .map_err(|e| NvcompError::LibraryLoadError(e.clone()))
}
#[cfg(test)]
mod tests {
    use crate::zstd;

    /// Test that we can call nvcompBatchedZstdDecompressGetTempSizeAsync.
    #[vortex_cuda_macros::test]
    fn test_get_decompress_temp_size() {
        let num_chunks = 10;
        let max_uncompressed_chunk_bytes = 65536; // 64KB recommended chunk size
        let max_total_uncompressed_bytes = num_chunks * max_uncompressed_chunk_bytes;

        let temp_bytes = zstd::get_decompress_temp_size(
            num_chunks,
            max_uncompressed_chunk_bytes,
            max_total_uncompressed_bytes,
        )
        .expect("get_decompress_temp_size failed");

        assert!(temp_bytes > 0, "Expected non-zero temp buffer size");
    }
}
