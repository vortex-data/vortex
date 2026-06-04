// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust wrappers around CUB DeviceScan operations used by CUDA kernels.

use std::ffi::c_void;

use crate::cub_library;
use crate::error::CubError;
use crate::error::check_cuda_error;
pub use crate::sys::cudaStream_t;

/// Get temporary storage size for CUB `DeviceScan::ExclusiveSum<i32>`.
pub fn exclusive_sum_i32_temp_size(num_items: i64) -> Result<usize, CubError> {
    let lib = cub_library()?;
    let mut temp_bytes: usize = 0;
    let err = unsafe { (lib.scan_exclusive_sum_i32_temp_size)(&raw mut temp_bytes, num_items) };
    check_cuda_error(err, "scan_exclusive_sum_i32_temp_size")?;
    Ok(temp_bytes)
}

/// Execute CUB `DeviceScan::ExclusiveSum<i32>`.
///
/// # Safety
///
/// All device pointers must be valid and properly sized:
/// - `d_temp` must have at least `temp_bytes` bytes allocated.
/// - `d_in` and `d_out` must have at least `num_items` `i32` values.
pub unsafe fn exclusive_sum_i32(
    d_temp: *mut c_void,
    temp_bytes: usize,
    d_in: *const i32,
    d_out: *mut i32,
    num_items: i64,
    stream: cudaStream_t,
) -> Result<(), CubError> {
    let lib = cub_library()?;
    let err =
        unsafe { (lib.scan_exclusive_sum_i32)(d_temp, temp_bytes, d_in, d_out, num_items, stream) };
    check_cuda_error(err, "scan_exclusive_sum_i32")
}

/// Get temporary storage size for CUB `DeviceScan::ExclusiveSum<i64>`.
pub fn exclusive_sum_i64_temp_size(num_items: i64) -> Result<usize, CubError> {
    let lib = cub_library()?;
    let mut temp_bytes: usize = 0;
    let err = unsafe { (lib.scan_exclusive_sum_i64_temp_size)(&raw mut temp_bytes, num_items) };
    check_cuda_error(err, "scan_exclusive_sum_i64_temp_size")?;
    Ok(temp_bytes)
}

/// Execute CUB `DeviceScan::ExclusiveSum<i64>`.
///
/// # Safety
///
/// All device pointers must be valid and properly sized:
/// - `d_temp` must have at least `temp_bytes` bytes allocated.
/// - `d_in` and `d_out` must have at least `num_items` `i64` values.
pub unsafe fn exclusive_sum_i64(
    d_temp: *mut c_void,
    temp_bytes: usize,
    d_in: *const i64,
    d_out: *mut i64,
    num_items: i64,
    stream: cudaStream_t,
) -> Result<(), CubError> {
    let lib = cub_library()?;
    let err =
        unsafe { (lib.scan_exclusive_sum_i64)(d_temp, temp_bytes, d_in, d_out, num_items, stream) };
    check_cuda_error(err, "scan_exclusive_sum_i64")
}
