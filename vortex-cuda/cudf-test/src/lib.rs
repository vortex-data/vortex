// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust bindings for cudf Arrow Device FFI operations.
//!
//! This crate provides a safe Rust interface to cudf's Arrow Device data
//! import functionality, allowing GPU data to be passed directly to cudf
//! for processing.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CStr;
use std::fmt;

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Error type for cudf operations
#[derive(Debug)]
pub struct CudfError {
    pub code: CudfErrorCode,
    pub message: String,
}

impl fmt::Display for CudfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudfError({:?}): {}", self.code, self.message)
    }
}

impl std::error::Error for CudfError {}

/// Result type for cudf operations
pub type Result<T> = std::result::Result<T, CudfError>;

/// Convert a CudfResult to a Rust Result
fn check_result(result: CudfResult) -> Result<()> {
    if result.code == CudfErrorCode_CUDF_SUCCESS {
        Ok(())
    } else {
        let message = if result.error_message.is_null() {
            format!("Unknown error (code: {:?})", result.code)
        } else {
            let msg = unsafe { CStr::from_ptr(result.error_message) }
                .to_string_lossy()
                .into_owned();
            // Free the error message
            unsafe { cudf_free_error(result.error_message) };
            msg
        };
        Err(CudfError {
            code: result.code,
            message,
        })
    }
}

/// Initialize the cudf/RMM runtime.
///
/// This must be called before any other cudf operations.
pub fn init() -> Result<()> {
    let result = unsafe { cudf_init() };
    check_result(result)
}

/// Load Arrow data from device memory into cudf.
///
/// # Safety
///
/// The schema and device_array must be valid Arrow C Data Interface structures
/// with device memory pointers.
pub unsafe fn load_from_arrow_device(
    schema: *const ArrowSchema,
    device_array: *const ArrowDeviceArray,
) -> Result<()> {
    let result = cudf_load_from_arrow_device(schema, device_array);
    check_result(result)
}

/// Load a single Arrow column from device memory into cudf.
///
/// # Safety
///
/// The schema and device_array must be valid Arrow C Data Interface structures
/// with device memory pointers.
pub unsafe fn load_column_from_arrow_device(
    schema: *const ArrowSchema,
    device_array: *const ArrowDeviceArray,
) -> Result<()> {
    let result = cudf_load_column_from_arrow_device(schema, device_array);
    check_result(result)
}

/// Get the number of rows in the loaded table.
pub fn get_row_count() -> Result<i64> {
    let mut count: i64 = 0;
    let result = unsafe { cudf_get_row_count(&mut count) };
    check_result(result)?;
    Ok(count)
}

/// Get the number of columns in the loaded table.
pub fn get_column_count() -> Result<i32> {
    let mut count: i32 = 0;
    let result = unsafe { cudf_get_column_count(&mut count) };
    check_result(result)?;
    Ok(count)
}

/// Count valid (non-null) values in a column.
pub fn count_valid(column_index: i32) -> Result<i64> {
    let mut count: i64 = 0;
    let result = unsafe { cudf_count_valid(column_index, &mut count) };
    check_result(result)?;
    Ok(count)
}

/// Sum values in an int64 column.
pub fn sum_int64(column_index: i32) -> Result<i64> {
    let mut sum: i64 = 0;
    let result = unsafe { cudf_sum_int64(column_index, &mut sum) };
    check_result(result)?;
    Ok(sum)
}

/// Free the currently loaded table.
pub fn free_table() -> Result<()> {
    let result = unsafe { cudf_free_table() };
    check_result(result)
}

/// RAII guard for the loaded table.
///
/// Automatically frees the table when dropped.
pub struct TableGuard;

impl Drop for TableGuard {
    fn drop(&mut self) {
        let _ = free_table();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() -> Result<()> {
        // This will fail if CUDA/cudf is not available, which is expected
        // in CI environments without GPU
        match init() {
            Ok(()) => {
                println!("cudf initialized successfully");
                Ok(())
            }
            Err(e) => {
                println!("cudf init failed (expected without GPU): {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_no_data_error() {
        // Without loading data, operations should fail with NO_DATA error
        let result = get_row_count();
        match result {
            Err(e) if e.code == CudfErrorCode_CUDF_ERROR_NO_DATA => {
                // Expected
            }
            Err(e) => {
                // Also acceptable - might fail for other reasons without GPU
                println!("Got error (acceptable): {}", e);
            }
            Ok(_) => {
                panic!("Expected error when no data loaded");
            }
        }
    }
}
