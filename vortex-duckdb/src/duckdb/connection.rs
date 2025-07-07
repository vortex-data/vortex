// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ptr;

use vortex::error::{VortexResult, vortex_err};

use crate::duckdb::{Database, QueryResult};
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

    /// Execute SQL query and return the result.
    pub fn query(&self, query: &str) -> VortexResult<QueryResult> {
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
                    CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
                }
            };

            unsafe { cpp::duckdb_destroy_result(&mut result) };
            return Err(vortex_err!("Failed to execute query: {}", error_msg));
        }

        Ok(unsafe { QueryResult::new(result) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_connection() -> VortexResult<Connection> {
        let db = Database::open_in_memory()?;
        db.connect()
    }

    #[test]
    fn test_connection_creation() {
        let conn = test_connection();
        assert!(conn.is_ok());
    }

    #[test]
    fn test_execute_success() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_invalid_sql() {
        let conn = test_connection().unwrap();
        let result = conn.query("INVALID SQL STATEMENT");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Failed to execute query"));
    }

    #[test]
    fn test_execute_with_null_bytes() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT\0 1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid query string"));
    }

    #[test]
    fn test_query_and_get_row_count_select() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 1, 2, 3").unwrap();
        assert_eq!(result.row_count().unwrap(), 1);
    }

    #[test]
    fn test_query_and_get_row_count_create_table() {
        let conn = test_connection().unwrap();

        // CREATE TABLE should return 0 rows
        let result = conn
            .query("CREATE TABLE test (id INTEGER, name VARCHAR)")
            .unwrap();
        assert_eq!(result.row_count().unwrap(), 0);
    }

    #[test]
    fn test_query_and_get_row_count_insert() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER, name VARCHAR)")
            .unwrap();

        let result = conn
            .query("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
            .unwrap();

        assert_eq!(result.row_count().unwrap(), 2);
    }

    #[test]
    fn test_query_invalid_sql() {
        let conn = test_connection().unwrap();
        let result = conn.query("INVALID SQL");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_single_value() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 42").unwrap();

        assert_eq!(result.column_count().unwrap(), 1);
        assert_eq!(result.row_count().unwrap(), 1);
        assert_eq!(result.get::<i64>(0, 0).unwrap(), 42);
    }

    #[test]
    fn test_query_multiple_rows() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER)").unwrap();
        conn.query("INSERT INTO test VALUES (1), (2), (3)").unwrap();

        let result = conn.query("SELECT id FROM test ORDER BY id").unwrap();

        assert_eq!(result.column_count().unwrap(), 1);
        assert_eq!(result.row_count().unwrap(), 3);
        assert_eq!(result.get::<i64>(0, 0).unwrap(), 1);
        assert_eq!(result.get::<i64>(0, 1).unwrap(), 2);
        assert_eq!(result.get::<i64>(0, 2).unwrap(), 3);
    }

    #[test]
    fn test_query_multiple_columns() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 1 as num, 'hello' as text").unwrap();

        assert_eq!(result.column_count().unwrap(), 2);
        assert_eq!(result.row_count().unwrap(), 1);
        assert_eq!(result.column_name(0).unwrap(), "num");
        assert_eq!(result.column_name(1).unwrap(), "text");
        assert_eq!(result.get::<i64>(0, 0).unwrap(), 1);
        assert_eq!(result.get::<String>(1, 0).unwrap(), "hello");
    }

    #[test]
    fn test_query_bounds_checking() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 1").unwrap();

        // Test row bounds
        assert!(result.get::<i64>(0, 1).is_err());

        // Test column bounds
        assert!(result.get::<i64>(1, 0).is_err());
    }

    #[test]
    fn test_query_column_types() {
        let conn = test_connection().unwrap();
        let result = conn
            .query("SELECT 1 as int_col, 'text' as str_col")
            .unwrap();

        assert_eq!(result.column_type(0), cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        assert_eq!(result.column_type(1), cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
    }

    #[test]
    fn test_null_handling() {
        let conn = test_connection().unwrap();
        let result = conn
            .query("SELECT NULL as null_col, 1 as not_null_col")
            .unwrap();

        assert!(result.is_null(0, 0).unwrap());
        assert!(!result.is_null(1, 0).unwrap());
    }

    #[test]
    fn test_type_conversion() {
        let conn = test_connection().unwrap();
        let result = conn
            .query("SELECT 42::TINYINT, 42::SMALLINT, 42::INTEGER, 42::BIGINT")
            .unwrap();

        assert_eq!(result.get::<i64>(0, 0).unwrap(), 42); // TINYINT -> i64
        assert_eq!(result.get::<i64>(1, 0).unwrap(), 42); // SMALLINT -> i64
        assert_eq!(result.get::<i64>(2, 0).unwrap(), 42); // INTEGER -> i64
        assert_eq!(result.get::<i64>(3, 0).unwrap(), 42); // BIGINT -> i64
    }

    #[test]
    fn test_query_and_get_row_count_update() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER, name VARCHAR)")
            .unwrap();
        conn.query("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')")
            .unwrap();

        let result = conn
            .query("UPDATE test SET name = 'Updated' WHERE id <= 2")
            .unwrap();
        assert_eq!(result.row_count().unwrap(), 2);
    }

    #[test]
    fn test_query_and_get_row_count_delete() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER)").unwrap();
        conn.query("INSERT INTO test VALUES (1), (2), (3)").unwrap();

        let result = conn.query("DELETE FROM test WHERE id > 1").unwrap();
        assert_eq!(result.row_count().unwrap(), 2);
    }
}
