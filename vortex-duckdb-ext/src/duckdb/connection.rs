use std::ptr;

use vortex::error::{VortexResult, vortex_err};

use crate::duckdb::Database;
use crate::{cpp, duckdb_try, wrapper};

wrapper!(
    /// A DuckDB connection.
    Connection,
    cpp::duckdb_connection,
    cpp::duckdb_disconnect
);

impl Connection {
    pub fn connect(db: &Database) -> VortexResult<Self> {
        let mut ptr: cpp::duckdb_connection = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_connect(db.as_ptr(), &mut ptr) },
            "Failed to connect to DuckDB database"
        );
        Ok(unsafe { Self::own(ptr) })
    }

    /// Execute SQL query and return the row count.
    pub fn execute_and_get_row_count(&self, query: &str) -> VortexResult<usize> {
        let mut result: cpp::duckdb_result = unsafe { std::mem::zeroed() };
        let query_cstr =
            std::ffi::CString::new(query).map_err(|_| vortex_err!("Invalid query string"))?;

        let status = unsafe { cpp::duckdb_query(self.as_ptr(), query_cstr.as_ptr(), &mut result) };

        if status != cpp::duckdb_state::DuckDBSuccess {
            let error_msg = unsafe {
                let error_ptr = cpp::duckdb_result_error(&mut result);
                if error_ptr.is_null() {
                    "Unknown DuckDB error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };

            unsafe { cpp::duckdb_destroy_result(&mut result) };
            return Err(vortex_err!("Failed to execute query: {}", error_msg));
        }

        let row_count = unsafe { cpp::duckdb_row_count(&mut result).try_into()? };
        unsafe { cpp::duckdb_destroy_result(&mut result) };

        Ok(row_count)
    }

    /// Execute SQL query.
    pub fn execute(&self, query: &str) -> VortexResult<()> {
        let mut result: cpp::duckdb_result = unsafe { std::mem::zeroed() };
        let query_cstr =
            std::ffi::CString::new(query).map_err(|_| vortex_err!("Invalid query string"))?;

        let status = unsafe { cpp::duckdb_query(self.as_ptr(), query_cstr.as_ptr(), &mut result) };

        if status != cpp::duckdb_state::DuckDBSuccess {
            let error_msg = unsafe {
                let error_ptr = cpp::duckdb_result_error(&mut result);
                if error_ptr.is_null() {
                    "Unknown DuckDB error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(error_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };

            unsafe { cpp::duckdb_destroy_result(&mut result) };
            return Err(vortex_err!("Failed to execute query: {}", error_msg));
        }

        unsafe { cpp::duckdb_destroy_result(&mut result) };

        Ok(())
    }
}
