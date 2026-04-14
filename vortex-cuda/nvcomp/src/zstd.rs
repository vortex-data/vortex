// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wrappers around nvcomp's batched ZSTD decompression API.

use std::ffi::c_void;

use crate::error::NvcompError;
use crate::error::check_status;
use crate::nvcomp_library;
use crate::sys;

/// Backend selection for nvcomp decompression.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecompressBackend {
    /// Let nvcomp auto-select the best backend for the hardware.
    #[default]
    Default,
    /// Use hardware decompression
    Hardware,
    /// Use CUDA
    Cuda,
}

impl DecompressBackend {
    fn to_nvcomp(self) -> sys::nvcompDecompressBackend_t {
        match self {
            Self::Default => sys::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
            Self::Hardware => sys::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_HARDWARE,
            Self::Cuda => sys::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_CUDA,
        }
    }
}

/// Options for batched ZSTD decompression.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZstdDecompressOpts {
    pub backend: DecompressBackend,
}

impl ZstdDecompressOpts {
    fn to_nvcomp(self) -> sys::nvcompBatchedZstdDecompressOpts_t {
        sys::nvcompBatchedZstdDecompressOpts_t {
            backend: self.backend.to_nvcomp(),
            reserved: [0; 60],
        }
    }
}

/// Computes required temporary buffer size for batched ZSTD decompression.
///
/// # Arguments
///
/// * `num_chunks` - Number of compressed chunks to decompress
/// * `max_uncompressed_chunk_bytes` - Maximum uncompressed size of any single chunk
/// * `max_total_uncompressed_bytes` - Total uncompressed size across all chunks
///
/// # Returns
///
/// The required size in bytes for the temporary buffer.
pub fn get_decompress_temp_size(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    max_total_uncompressed_bytes: usize,
) -> Result<usize, NvcompError> {
    get_decompress_temp_size_with_opts(
        num_chunks,
        max_uncompressed_chunk_bytes,
        max_total_uncompressed_bytes,
        ZstdDecompressOpts::default(),
    )
}

/// Computes required temporary buffer size with custom options.
///
/// # Arguments
///
/// * `num_chunks` - Number of compressed chunks to decompress
/// * `max_uncompressed_chunk_bytes` - Maximum uncompressed size of any single chunk
/// * `max_total_uncompressed_bytes` - Total uncompressed size across all chunks
/// * `opts` - Decompression options
///
/// # Returns
///
/// The required size in bytes for the temporary buffer.
pub fn get_decompress_temp_size_with_opts(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    max_total_uncompressed_bytes: usize,
    opts: ZstdDecompressOpts,
) -> Result<usize, NvcompError> {
    let library = nvcomp_library()?;

    let mut temp_bytes: usize = 0;

    let status = unsafe {
        library.nvcompBatchedZstdDecompressGetTempSizeAsync(
            num_chunks,
            max_uncompressed_chunk_bytes,
            opts.to_nvcomp(),
            &raw mut temp_bytes,
            max_total_uncompressed_bytes,
        )
    };

    check_status(status)?;
    Ok(temp_bytes)
}

/// Launches batched ZSTD decompression asynchronously on the GPU.
///
/// This function decompresses multiple ZSTD-compressed chunks in parallel on the GPU.
/// All pointer arguments must point to device memory, and the operation is executed
/// asynchronously on the provided CUDA stream.
///
/// # Arguments
///
/// * `device_compressed_ptrs` - Device pointer to array of pointers to compressed chunks
/// * `device_compressed_bytes` - Device pointer to array of compressed chunk sizes
/// * `device_uncompressed_bytes` - Device pointer to array of expected uncompressed sizes
/// * `device_actual_uncompressed_bytes` - Device pointer to array for actual uncompressed sizes (output)
/// * `num_chunks` - Number of chunks to decompress
/// * `device_temp_ptr` - Device pointer to temporary workspace buffer
/// * `temp_bytes` - Size of temporary buffer in bytes
/// * `device_uncompressed_ptrs` - Device pointer to array of pointers to output buffers
/// * `device_statuses` - Device pointer to array for per-chunk status codes (output)
/// * `stream` - CUDA stream to execute on
///
/// # Safety
///
/// - All device pointers must be valid and point to properly allocated device memory
/// - `device_compressed_ptrs` must point to valid device pointers
/// - `device_uncompressed_ptrs` must point to valid device pointers
/// - Each output buffer must have at least the corresponding `device_uncompressed_bytes` size
/// - `device_temp_ptr` must have at least `temp_bytes` allocated
/// - The stream must be valid
#[expect(clippy::too_many_arguments)]
pub unsafe fn decompress_async(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    num_chunks: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_statuses: *mut sys::nvcompStatus_t,
    stream: sys::cudaStream_t,
) -> Result<(), NvcompError> {
    // SAFETY: Caller has to ensure all pointers are valid.
    unsafe {
        decompress_async_with_opts(
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            num_chunks,
            device_temp_ptr,
            temp_bytes,
            device_uncompressed_ptrs,
            device_statuses,
            stream,
            ZstdDecompressOpts::default(),
        )
    }
}

/// Launches batched ZSTD decompression asynchronously with custom options.
///
/// # Safety
///
/// Same requirements as [`decompress_async`].
#[expect(clippy::too_many_arguments)]
pub unsafe fn decompress_async_with_opts(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    num_chunks: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_statuses: *mut sys::nvcompStatus_t,
    stream: sys::cudaStream_t,
    opts: ZstdDecompressOpts,
) -> Result<(), NvcompError> {
    let library = nvcomp_library()?;

    let status = unsafe {
        library.nvcompBatchedZstdDecompressAsync(
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            num_chunks,
            device_temp_ptr,
            temp_bytes,
            device_uncompressed_ptrs,
            opts.to_nvcomp(),
            device_statuses,
            stream,
        )
    };

    check_status(status)
}
