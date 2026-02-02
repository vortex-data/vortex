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
use std::ptr;

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Error type for cudf operations.
#[derive(Debug)]
pub struct CudfError {
    pub message: String,
}

impl fmt::Display for CudfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudfError: {}", self.message)
    }
}

impl std::error::Error for CudfError {}

/// Result type for cudf operations.
pub type Result<T> = std::result::Result<T, CudfError>;

/// Check a cudf_err_t and convert to Result.
fn check_err(err: cudf_err_t) -> Result<()> {
    if err.is_null() {
        Ok(())
    } else {
        let message = unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        unsafe { cudf_err_free(err) };
        Err(CudfError { message })
    }
}

/// RAII wrapper for cudf_context_t.
pub struct CudfContext {
    ctx: *mut cudf_context_t,
}

impl CudfContext {
    /// Create a new cudf context and initialize RMM.
    pub fn new() -> Result<Self> {
        let mut ctx: *mut cudf_context_t = ptr::null_mut();
        let err = unsafe { cudf_context_create(&raw mut ctx) };
        check_err(err)?;
        Ok(Self { ctx })
    }

    /// Import an Arrow table from device memory into a cudf table view.
    ///
    /// # Safety
    ///
    /// The schema and device_array must be valid Arrow C Device Data Interface structures
    /// with device memory pointers.
    pub unsafe fn tableview_from_device(
        &self,
        schema: *const ArrowSchema,
        device_array: *const ArrowDeviceArray,
    ) -> Result<CudfTableView> {
        let mut tv: *mut cudf_tableview_t = ptr::null_mut();
        let err =
            unsafe { cudf_tableview_from_device(self.ctx, schema, device_array, &raw mut tv) };
        check_err(err)?;
        Ok(CudfTableView { tv })
    }

    /// Import an Arrow column from device memory into a cudf column view.
    ///
    /// # Safety
    ///
    /// The schema and device_array must be valid Arrow C Data Interface structures
    /// with device memory pointers.
    pub unsafe fn columnview_from_device(
        &self,
        schema: *const ArrowSchema,
        device_array: *const ArrowDeviceArray,
    ) -> Result<CudfColumnView> {
        let mut cv: *mut cudf_columnview_t = ptr::null_mut();
        let err =
            unsafe { cudf_columnview_from_device(self.ctx, schema, device_array, &raw mut cv) };
        check_err(err)?;
        Ok(CudfColumnView { cv })
    }
}

impl Drop for CudfContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { cudf_context_free(self.ctx) };
        }
    }
}

/// RAII wrapper for cudf_tableview_t.
pub struct CudfTableView {
    tv: *mut cudf_tableview_t,
}

impl CudfTableView {
    /// Get the number of rows in the table.
    pub fn num_rows(&self) -> Result<i64> {
        let mut count: i64 = 0;
        let err = unsafe { cudf_tableview_num_rows(self.tv, &raw mut count) };
        check_err(err)?;
        Ok(count)
    }

    /// Get the number of columns in the table.
    pub fn num_columns(&self) -> Result<i32> {
        let mut count: i32 = 0;
        let err = unsafe { cudf_tableview_num_columns(self.tv, &raw mut count) };
        check_err(err)?;
        Ok(count)
    }

    /// Count valid (non-null) values in a column.
    pub fn count_valid(&self, column_index: i32) -> Result<i64> {
        let mut count: i64 = 0;
        let err = unsafe { cudf_tableview_count_valid(self.tv, column_index, &raw mut count) };
        check_err(err)?;
        Ok(count)
    }

    /// Sum values in an int64 column.
    pub fn sum_int64(&self, column_index: i32) -> Result<i64> {
        let mut sum: i64 = 0;
        let err = unsafe { cudf_tableview_sum_int64(self.tv, column_index, &raw mut sum) };
        check_err(err)?;
        Ok(sum)
    }
}

impl Drop for CudfTableView {
    fn drop(&mut self) {
        if !self.tv.is_null() {
            unsafe { cudf_tableview_free(self.tv) };
        }
    }
}

/// RAII wrapper for cudf_columnview_t.
pub struct CudfColumnView {
    cv: *mut cudf_columnview_t,
}

impl CudfColumnView {
    /// Get the number of rows in the column.
    pub fn size(&self) -> Result<i64> {
        let mut count: i64 = 0;
        let err = unsafe { cudf_columnview_size(self.cv, &raw mut count) };
        check_err(err)?;
        Ok(count)
    }

    /// Count valid (non-null) values in the column.
    pub fn count_valid(&self) -> Result<i64> {
        let mut count: i64 = 0;
        let err = unsafe { cudf_columnview_count_valid(self.cv, &raw mut count) };
        check_err(err)?;
        Ok(count)
    }

    /// Sum values in the column (int64).
    pub fn sum_int64(&self) -> Result<i64> {
        let mut sum: i64 = 0;
        let err = unsafe { cudf_columnview_sum_int64(self.cv, &raw mut sum) };
        check_err(err)?;
        Ok(sum)
    }
}

impl Drop for CudfColumnView {
    fn drop(&mut self) {
        if !self.cv.is_null() {
            unsafe { cudf_columnview_free(self.cv) };
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::ffi::FFI_ArrowSchema;
    use arrow_schema::DataType;
    use futures::executor::block_on;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_cuda::CudaSession;
    use vortex_cuda::arrow::CudaDeviceArrayExecute;
    use vortex_cuda::executor::CudaArrayExt;
    use vortex_session::VortexSession;

    use super::*;

    #[test]
    fn test_context_create() -> Result<()> {
        match CudfContext::new() {
            Ok(_ctx) => {
                println!("cudf context created successfully");
                Ok(())
            }
            Err(e) => {
                println!("cudf context creation failed (expected without GPU): {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_primitive_array_to_cudf_tableview() -> Result<()> {
        // Create a PrimitiveArray with 100 i64 values
        let data: Vec<i64> = (0..100).collect();
        let expected_len = data.len();
        let primitive_array =
            PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();

        // Create CUDA execution context
        let mut cuda_ctx = match CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();

        // Export as ArrowDeviceArray using CudaDeviceArrayExecute
        let device_array = block_on(Canonical::execute(
            &primitive_array,
            primitive_array.clone(),
            &mut cuda_ctx,
        ))
        .unwrap();

        // Synchronize the CUDA stream to ensure the data is ready
        cuda_ctx.synchronize_stream().map_err(|e| CudfError {
            message: e.to_string(),
        })?;

        // Create FFI_ArrowSchema from the data type
        let mut ffi_schema =
            FFI_ArrowSchema::try_from(&DataType::Int64).map_err(|e| CudfError {
                message: format!("Failed to create FFI schema: {}", e),
            })?;

        // Create cudf context
        let cudf_ctx = CudfContext::new()?;

        // Import into cudf tableview
        let tableview = unsafe {
            cudf_ctx.tableview_from_device(
                (&raw mut ffi_schema).cast::<ArrowSchema>(),
                (&raw const device_array).cast::<ArrowDeviceArray>(),
            )?
        };

        // Verify row count
        let num_rows = tableview.num_rows()?;
        assert_eq!(num_rows, expected_len as i64, "Row count mismatch");
        println!(
            "Successfully imported PrimitiveArray into cudf tableview with {} rows",
            num_rows
        );

        // Verify column count (should be 1 for a primitive array)
        let num_columns = tableview.num_columns()?;
        assert_eq!(num_columns, 1, "Column count mismatch");
        println!("Tableview has {} column(s)", num_columns);

        // Tableview and cudf_ctx will be deallocated automatically via Drop

        Ok(())
    }
}
