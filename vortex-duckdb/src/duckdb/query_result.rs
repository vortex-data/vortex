// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;

use bitvec::macros::internal::funty::Fundamental;
use vortex::error::{VortexResult, vortex_err};

use crate::{cpp, wrapper};

wrapper! {
    /// A wrapper around a DuckDB query result.
    #[derive(Debug)]
    QueryResult,
    *mut cpp::duckdb_result,
    |ptr: &mut *mut cpp::duckdb_result| {
        if !ptr.is_null() {
            unsafe {
                cpp::duckdb_destroy_result(&mut **ptr);
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

    /// Check bounds for row and column indices.
    fn check_bounds(&self, col_idx: usize, row_idx: usize) -> VortexResult<()> {
        if row_idx >= self.row_count()? {
            return Err(vortex_err!("Row index {} out of bounds", row_idx));
        }
        if col_idx >= self.column_count()? {
            return Err(vortex_err!("Column index {} out of bounds", col_idx));
        }
        Ok(())
    }

    /// Get the number of columns in the result.
    pub fn column_count(&self) -> VortexResult<usize> {
        unsafe {
            usize::try_from(cpp::duckdb_column_count(self.as_ptr()))
                .map_err(|_| vortex_err!("Column count too large to fit in usize"))
        }
    }

    /// Get the number of rows in the result for SELECT operations,
    /// or the number of affected rows for (INSERT/UPDATE/DELETE) operations.
    pub fn row_count(&self) -> VortexResult<usize> {
        unsafe {
            let rows_changed = cpp::duckdb_rows_changed(self.as_ptr());
            if rows_changed > 0 {
                // (INSERT, UPDATE, DELETE) - return affected rows
                usize::try_from(rows_changed)
                    .map_err(|_| vortex_err!("Rows changed count too large to fit in usize"))
            } else {
                // SELECT - return result row count
                usize::try_from(cpp::duckdb_row_count(self.as_ptr()))
                    .map_err(|_| vortex_err!("Row count too large to fit in usize"))
            }
        }
    }

    /// Get the name of a column by index.
    pub fn column_name(&self, col_idx: usize) -> VortexResult<&str> {
        unsafe {
            let name_ptr = cpp::duckdb_column_name(self.as_ptr(), col_idx as u64);
            if name_ptr.is_null() {
                return Err(vortex_err!("Invalid column index: {}", col_idx));
            }
            CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|_| vortex_err!("Invalid UTF-8 in column name"))
        }
    }

    /// Get the type of a column by index.
    pub fn column_type(&self, col_idx: usize) -> cpp::DUCKDB_TYPE {
        unsafe { cpp::duckdb_column_type(self.as_ptr(), col_idx as u64) }
    }

    /// Try to get a value at the specified column and row index for the specified type.
    pub fn get<T>(&self, col_idx: usize, row_idx: usize) -> VortexResult<T>
    where
        T: TryFrom<QueryResultCell, Error = vortex::error::VortexError>,
    {
        self.check_bounds(col_idx, row_idx)?;

        let value = unsafe { QueryResultCell::new(self.as_ptr(), col_idx, row_idx) };

        T::try_from(value)
    }

    pub fn cell_as_str(&self, col_idx: usize, row_idx: usize) -> VortexResult<String> {
        self.check_bounds(col_idx, row_idx)?;

        unsafe { QueryResultCell::new(self.as_ptr(), col_idx, row_idx) }.to_string()
    }

    /// Check if a value is null.
    pub fn is_null(&self, col_idx: usize, row_idx: usize) -> VortexResult<bool> {
        self.check_bounds(col_idx, row_idx)?;

        // Get nullmask data for the column
        unsafe {
            let nullmask_ptr = cpp::duckdb_nullmask_data(self.as_ptr(), col_idx as u64);
            if nullmask_ptr.is_null() {
                // No nullmask means no nulls
                return Ok(false);
            }

            // Access the bool array directly
            let is_null = *nullmask_ptr.add(row_idx);
            Ok(is_null)
        }
    }
}

/// A result cell returned by a DuckDB query.
pub struct QueryResultCell {
    result_ptr: *mut cpp::duckdb_result,
    col_idx: u64,
    row_idx: u64,
    column_type: cpp::DUCKDB_TYPE,
}

impl QueryResultCell {
    unsafe fn new(result_ptr: *mut cpp::duckdb_result, col_idx: usize, row_idx: usize) -> Self {
        let column_type = unsafe { cpp::duckdb_column_type(result_ptr, col_idx as u64) };
        Self {
            result_ptr,
            col_idx: col_idx as u64,
            row_idx: row_idx as u64,
            column_type,
        }
    }

    pub fn column_type(&self) -> cpp::DUCKDB_TYPE {
        self.column_type
    }

    pub fn to_string(&self) -> VortexResult<String> {
        let slice = unsafe {
            let d = cpp::duckdb_value_string(self.result_ptr, self.col_idx, self.row_idx);
            std::slice::from_raw_parts(d.data as *const u8, d.size.as_usize())
        };
        String::from_utf8(slice.to_vec()).map_err(|e| vortex_err!("{e}"))
    }
}

// TODO(Alex): Revisit the marshalling logic to consider going through arrow by
// calling `duckdb_query_arrow`. Most likely the marshalling code below will go away.

impl TryFrom<QueryResultCell> for i32 {
    type Error = vortex::error::VortexError;

    fn try_from(entry: QueryResultCell) -> Result<Self, Self::Error> {
        match entry.column_type {
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => Ok(unsafe {
                cpp::duckdb_value_int32(entry.result_ptr, entry.col_idx, entry.row_idx)
            }),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
                let val = unsafe {
                    cpp::duckdb_value_int8(entry.result_ptr, entry.col_idx, entry.row_idx)
                };
                Ok(val as i32)
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
                let val = unsafe {
                    cpp::duckdb_value_int16(entry.result_ptr, entry.col_idx, entry.row_idx)
                };
                Ok(val as i32)
            }
            _ => Err(vortex_err!("Cannot convert {:?} to i32", entry.column_type)),
        }
    }
}

impl TryFrom<QueryResultCell> for i64 {
    type Error = vortex::error::VortexError;

    fn try_from(entry: QueryResultCell) -> Result<Self, Self::Error> {
        match entry.column_type {
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => unsafe {
                let val = cpp::duckdb_value_hugeint(entry.result_ptr, entry.col_idx, entry.row_idx);
                // Convert hugeint to i64 - use lower part if it fits, otherwise error.
                if val.upper == 0 || (val.upper == -1 && (val.lower as i64) < 0) {
                    Ok(val.lower as i64)
                } else {
                    Err(vortex_err!("HUGEINT value too large to fit in i64"))
                }
            },
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => Ok(unsafe {
                cpp::duckdb_value_int64(entry.result_ptr, entry.col_idx, entry.row_idx)
            }),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
                let val = unsafe {
                    cpp::duckdb_value_int32(entry.result_ptr, entry.col_idx, entry.row_idx)
                };
                Ok(val as i64)
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
                let val = unsafe {
                    cpp::duckdb_value_int8(entry.result_ptr, entry.col_idx, entry.row_idx)
                };
                Ok(val as i64)
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
                let val = unsafe {
                    cpp::duckdb_value_int16(entry.result_ptr, entry.col_idx, entry.row_idx)
                };
                Ok(val as i64)
            }
            _ => Err(vortex_err!("Cannot convert {:?} to i64", entry.column_type)),
        }
    }
}

impl TryFrom<QueryResultCell> for f32 {
    type Error = vortex::error::VortexError;

    fn try_from(entry: QueryResultCell) -> Result<Self, Self::Error> {
        match entry.column_type {
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => Ok(unsafe {
                cpp::duckdb_value_float(entry.result_ptr, entry.col_idx, entry.row_idx)
            }),
            _ => Err(vortex_err!("Cannot convert {:?} to f32", entry.column_type)),
        }
    }
}

impl TryFrom<QueryResultCell> for f64 {
    type Error = vortex::error::VortexError;

    fn try_from(entry: QueryResultCell) -> Result<Self, Self::Error> {
        match entry.column_type {
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => Ok(unsafe {
                cpp::duckdb_value_double(entry.result_ptr, entry.col_idx, entry.row_idx)
            }),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
                let val = unsafe {
                    cpp::duckdb_value_float(entry.result_ptr, entry.col_idx, entry.row_idx)
                };
                Ok(val as f64)
            }
            _ => Err(vortex_err!("Cannot convert {:?} to f64", entry.column_type)),
        }
    }
}

impl TryFrom<QueryResultCell> for String {
    type Error = vortex::error::VortexError;

    fn try_from(entry: QueryResultCell) -> Result<Self, Self::Error> {
        match entry.column_type {
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR => unsafe {
                let str_ptr =
                    cpp::duckdb_value_varchar(entry.result_ptr, entry.col_idx, entry.row_idx);
                if str_ptr.is_null() {
                    return Ok(String::new());
                }
                let c_str = CStr::from_ptr(str_ptr);
                let result = c_str.to_string_lossy().into_owned();
                cpp::duckdb_free(str_ptr as *mut std::ffi::c_void);
                Ok(result)
            },
            _ => Err(vortex_err!(
                "Cannot convert {:?} to String",
                entry.column_type
            )),
        }
    }
}
