// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust bindings to NVIDIA nvCOMP compression library.
//!
//! This crate provides raw FFI bindings to nvCOMP, generated via bindgen
//! from the nvCOMP C headers. The nvCOMP SDK is automatically downloaded
//! at build time.
//!
//! # Platform Support
//!
//! nvCOMP is only available on Linux x86_64 and ARM64. On other platforms,
//! this crate still builds against the CUDA APIs but can't be run.
//!
//! # Runtime Requirements
//!
//! The nvcomp library is linked dynamically.

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
