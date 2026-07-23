// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust wrappers around CUB DeviceScan operations used by CUDA kernels.

use std::ffi::c_void;

use crate::cub_library;
use crate::error::CubError;
use crate::error::check_cuda_error;
pub use crate::sys::cudaStream_t;

/// Element type of the `lengths` input to [`exclusive_sum_lengths`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthType {
    /// `u8` lengths.
    U8,
    /// `i8` lengths.
    I8,
    /// `u16` lengths.
    U16,
    /// `i16` lengths.
    I16,
    /// `u32` lengths.
    U32,
    /// `i32` lengths.
    I32,
    /// `u64` lengths.
    U64,
    /// `i64` lengths.
    I64,
}

/// Get temporary storage size for the fused widen + exclusive-sum scan
/// ([`exclusive_sum_lengths`]).
pub fn exclusive_sum_lengths_temp_size(
    ty: LengthType,
    num_offsets: i64,
) -> Result<usize, CubError> {
    let lib = cub_library()?;
    let mut temp_bytes: usize = 0;
    let err = unsafe {
        match ty {
            LengthType::U8 => {
                (lib.scan_exclusive_sum_lengths_u8_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::I8 => {
                (lib.scan_exclusive_sum_lengths_i8_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::U16 => {
                (lib.scan_exclusive_sum_lengths_u16_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::I16 => {
                (lib.scan_exclusive_sum_lengths_i16_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::U32 => {
                (lib.scan_exclusive_sum_lengths_u32_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::I32 => {
                (lib.scan_exclusive_sum_lengths_i32_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::U64 => {
                (lib.scan_exclusive_sum_lengths_u64_temp_size)(&raw mut temp_bytes, num_offsets)
            }
            LengthType::I64 => {
                (lib.scan_exclusive_sum_lengths_i64_temp_size)(&raw mut temp_bytes, num_offsets)
            }
        }
    };
    check_cuda_error(err, "scan_exclusive_sum_lengths_temp_size")?;
    Ok(temp_bytes)
}

/// Execute the fused widen + CUB `DeviceScan::ExclusiveSum` over per-row
/// lengths: scans `num_offsets` (= `num_rows + 1`) values where value `i` is
/// `lengths[i]` widened to u64 and the final slot contributes zero, so
/// `d_out[num_offsets - 1]` is the total. A negative length (signed types
/// only) raises `*status` to 2 and contributes zero bytes.
///
/// # Safety
///
/// All device pointers must be valid and properly sized:
/// - `d_temp` must have at least `temp_bytes` bytes allocated.
/// - `lengths` must have at least `num_offsets - 1` elements of type `ty`.
/// - `d_out` must have at least `num_offsets` `u64` values.
/// - `status` must point to a valid device `u32`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn exclusive_sum_lengths(
    ty: LengthType,
    d_temp: *mut c_void,
    temp_bytes: usize,
    lengths: *const c_void,
    d_out: *mut u64,
    status: *mut u32,
    num_offsets: i64,
    stream: cudaStream_t,
) -> Result<(), CubError> {
    let lib = cub_library()?;
    let err = unsafe {
        match ty {
            LengthType::U8 => (lib.scan_exclusive_sum_lengths_u8)(
                d_temp,
                temp_bytes,
                lengths as *const u8,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::I8 => (lib.scan_exclusive_sum_lengths_i8)(
                d_temp,
                temp_bytes,
                lengths as *const i8,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::U16 => (lib.scan_exclusive_sum_lengths_u16)(
                d_temp,
                temp_bytes,
                lengths as *const u16,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::I16 => (lib.scan_exclusive_sum_lengths_i16)(
                d_temp,
                temp_bytes,
                lengths as *const i16,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::U32 => (lib.scan_exclusive_sum_lengths_u32)(
                d_temp,
                temp_bytes,
                lengths as *const u32,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::I32 => (lib.scan_exclusive_sum_lengths_i32)(
                d_temp,
                temp_bytes,
                lengths as *const i32,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::U64 => (lib.scan_exclusive_sum_lengths_u64)(
                d_temp,
                temp_bytes,
                lengths as *const u64,
                d_out,
                status,
                num_offsets,
                stream,
            ),
            LengthType::I64 => (lib.scan_exclusive_sum_lengths_i64)(
                d_temp,
                temp_bytes,
                lengths as *const i64,
                d_out,
                status,
                num_offsets,
                stream,
            ),
        }
    };
    check_cuda_error(err, "scan_exclusive_sum_lengths")
}

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
