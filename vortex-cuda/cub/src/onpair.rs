// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust wrappers around the fused OnPair per-batch offsets regeneration.

use std::ffi::c_void;

use crate::cub_library;
use crate::error::CubError;
use crate::error::check_cuda_error;
pub use crate::sys::cudaStream_t;

/// Get temporary storage size (the look-back tile state) for
/// [`batch_offsets`].
pub fn batch_offsets_temp_size(num_batches: i64) -> Result<usize, CubError> {
    let lib = cub_library()?;
    let mut temp_bytes: usize = 0;
    let err = unsafe { (lib.onpair_batch_offsets_temp_size)(&raw mut temp_bytes, num_batches) };
    check_cuda_error(err, "onpair_batch_offsets_temp_size")?;
    Ok(temp_bytes)
}

/// Regenerate the OnPair decode kernel's per-batch output offsets in one fused
/// sweep: each warp reduces one 128-token batch's decoded size and the
/// exclusive scan over the sizes runs in-kernel via decoupled look-back.
/// Writes `num_batches + 1` offsets; the last is the total decoded byte count.
/// `code_width` selects the code stream's element size in bytes (1 or 2). A
/// code outside the dictionary raises `*status` to 1 and contributes zero
/// bytes.
///
/// # Safety
///
/// All device pointers must be valid and properly sized:
/// - `d_temp` must have at least `temp_bytes` bytes allocated.
/// - `codes` must have at least `total_tokens` elements of `code_width` bytes
///   each, with `total_tokens <= num_batches * 128`.
/// - `lens` must have at least `dict_size` bytes.
/// - `chunk_offsets` must have at least `num_batches + 1` `u64` values.
/// - `status` must point to a valid device `u32`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_offsets(
    d_temp: *mut c_void,
    temp_bytes: usize,
    codes: *const c_void,
    code_width: u32,
    lens: *const u8,
    dict_size: u32,
    total_tokens: u64,
    chunk_offsets: *mut u64,
    status: *mut u32,
    num_batches: i64,
    stream: cudaStream_t,
) -> Result<(), CubError> {
    let lib = cub_library()?;
    let err = unsafe {
        (lib.onpair_batch_offsets)(
            d_temp,
            temp_bytes,
            codes,
            code_width,
            lens,
            dict_size,
            total_tokens,
            chunk_offsets,
            status,
            num_batches,
            stream,
        )
    };
    check_cuda_error(err, "onpair_batch_offsets")
}
