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

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::Library;
use libloading::Symbol;

/// Raw FFI type definitions from bindgen (no function declarations).
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all
)]
pub mod sys;

mod error;
pub mod zstd;

pub use error::NvcompError;

/// The loaded nvcomp library instance.
static NVCOMP_LIB: OnceLock<Result<NvcompLibrary, String>> = OnceLock::new();

/// A loaded nvcomp library with function pointers.
pub struct NvcompLibrary {
    // Keep the library handle alive
    #[allow(dead_code)]
    library: Library,

    // Function pointers
    pub(crate) batched_zstd_decompress_get_temp_size_async: BatchedZstdDecompressGetTempSizeAsyncFn,
    pub(crate) batched_zstd_decompress_async: BatchedZstdDecompressAsyncFn,
}

// Function pointer types matching the nvcomp C API
type BatchedZstdDecompressGetTempSizeAsyncFn = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    opts: sys::nvcompBatchedZstdDecompressOpts_t,
    temp_bytes: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> sys::nvcompStatus_t;

type BatchedZstdDecompressAsyncFn = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptr: *const *mut c_void,
    opts: sys::nvcompBatchedZstdDecompressOpts_t,
    device_statuses: *mut sys::nvcompStatus_t,
    stream: sys::cudaStream_t,
) -> sys::nvcompStatus_t;

impl NvcompLibrary {
    /// Loads the nvcomp library from the given path.
    ///
    /// # Safety
    ///
    /// The library at the given path must be a valid nvcomp shared library.
    unsafe fn load_from_path(lib_path: &std::path::Path) -> Result<Self, String> {
        let library = unsafe {
            Library::new(lib_path).map_err(|e| format!("Failed to load nvcomp library: {e}"))?
        };

        // Load function pointers
        let batched_zstd_decompress_get_temp_size_async = unsafe {
            let sym: Symbol<BatchedZstdDecompressGetTempSizeAsyncFn> = library
                .get(b"nvcompBatchedZstdDecompressGetTempSizeAsync\0")
                .map_err(|e| {
                    format!("Failed to load nvcompBatchedZstdDecompressGetTempSizeAsync: {e}")
                })?;
            *sym
        };

        let batched_zstd_decompress_async = unsafe {
            let sym: Symbol<BatchedZstdDecompressAsyncFn> = library
                .get(b"nvcompBatchedZstdDecompressAsync\0")
                .map_err(|e| format!("Failed to load nvcompBatchedZstdDecompressAsync: {e}"))?;
            *sym
        };

        Ok(Self {
            library,
            batched_zstd_decompress_get_temp_size_async,
            batched_zstd_decompress_async,
        })
    }

    fn load_nvcomp() -> Result<Self, String> {
        let lib_name = "libnvcomp.so";
        let build_lib_dir = env!("OUT_DIR");
        let sdk_lib_path = PathBuf::from(build_lib_dir)
            .join("nvcomp-sdk")
            .join("lib")
            .join(lib_name);

        unsafe { Self::load_from_path(&sdk_lib_path) }
    }
}

/// Gets a reference to the loaded nvcomp library.
///
/// The library is loaded lazily on first access. Returns an error if the
/// library cannot be found or loaded.
pub fn nvcomp_library() -> Result<&'static NvcompLibrary, NvcompError> {
    NVCOMP_LIB
        .get_or_init(NvcompLibrary::load_nvcomp)
        .as_ref()
        .map_err(|e| NvcompError::LibraryLoadError(e.clone()))
}

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use crate::zstd;

    /// Test that we can call nvcompBatchedZstdDecompressGetTempSizeAsync.
    #[test]
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
