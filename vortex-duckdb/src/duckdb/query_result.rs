// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::cpp;
use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::DataChunk;
use crate::lifetime_wrapper;

lifetime_wrapper! {
    /// A wrapper around a DuckDB query result.
    #[derive(Debug)]
    QueryResult,
    *mut cpp::duckdb_result,
    |ptr: &mut *mut cpp::duckdb_result| {
        if !ptr.is_null() {
            unsafe {
                cpp::duckdb_destroy_result(&raw mut **ptr);
                drop(Box::from_raw(*ptr));
            }
        }
    }
}

impl QueryResult {
    /// Create a new `QueryResult` from a `duckdb_result`.
    ///
    /// Takes ownership of the result and will destroy it on drop.
    pub unsafe fn new(result: cpp::duckdb_result) -> Self {
        let boxed = Box::new(result);
        unsafe { Self::own(Box::into_raw(boxed)) }
    }
}

impl QueryResultRef {
    /// Get the number of columns in the result.
    pub fn column_count(&self) -> u64 {
        unsafe { cpp::duckdb_column_count(self.as_ptr()) }
    }

    /// Get the number of rows in the result for SELECT operations,
    /// or the number of affected rows for (INSERT/UPDATE/DELETE) operations.
    pub fn row_count(&self) -> u64 {
        let rows_changed = unsafe { cpp::duckdb_rows_changed(self.as_ptr()) };
        if rows_changed > 0 {
            // (INSERT, UPDATE, DELETE) - return affected rows
            rows_changed
        } else {
            // SELECT - return result row count
            unsafe { cpp::duckdb_row_count(self.as_ptr()) }
        }
    }

    /// Get the name of a column by index.
    pub fn column_name(&self, col_idx: usize) -> VortexResult<&str> {
        unsafe {
            let name_ptr = cpp::duckdb_column_name(self.as_ptr(), col_idx as u64);
            if name_ptr.is_null() {
                vortex_bail!("Invalid column index: {}", col_idx);
            }
            CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| vortex_err!("Invalid UTF-8 in column name"))
        }
    }

    /// Get the type of a column by index.
    pub fn column_type(&self, col_idx: usize) -> LogicalType {
        let dtype = unsafe { cpp::duckdb_column_type(self.as_ptr(), col_idx as u64) };
        if dtype == DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL {
            let lt = unsafe { cpp::duckdb_column_logical_type(self.as_ptr(), col_idx as u64) };
            let precision = unsafe { cpp::duckdb_decimal_width(lt) };
            let scale = unsafe { cpp::duckdb_decimal_scale(lt) };

            LogicalType::decimal_type(precision, scale).vortex_expect("valid decimal")
        } else {
            LogicalType::new(dtype)
        }
    }
}

use crate::duckdb::LogicalType;

impl IntoIterator for QueryResult {
    type Item = DataChunk;
    type IntoIter = QueryResultIter;

    fn into_iter(self) -> Self::IntoIter {
        QueryResultIter::new(self)
    }
}

pub struct QueryResultIter {
    result: QueryResult,
}

impl QueryResultIter {
    pub fn new(result: QueryResult) -> Self {
        Self { result }
    }
}

impl Iterator for QueryResultIter {
    type Item = DataChunk;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = unsafe { cpp::duckdb_fetch_chunk(*self.result.as_ptr()) };
        if chunk.is_null() {
            return None;
        }
        Some(unsafe { DataChunk::own(chunk) })
    }
}
