// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ptr;

use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::ClientContext;
use crate::duckdb::ClientContextRef;
use crate::duckdb::DatabaseRef;
use crate::duckdb::QueryResult;
use crate::duckdb_try;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A DuckDB connection.
    Connection,
    cpp::duckdb_connection,
    cpp::duckdb_disconnect
);

impl Connection {
    pub fn connect(db: &DatabaseRef) -> VortexResult<Self> {
        let mut ptr: cpp::duckdb_connection = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_connect(db.as_ptr(), &raw mut ptr) },
            "Failed to connect to DuckDB database"
        );
        Ok(unsafe { Self::own(ptr) })
    }
}

impl ConnectionRef {
    /// Execute SQL query and return the result.
    pub fn query(&self, query: &str) -> VortexResult<QueryResult> {
        let mut result: cpp::duckdb_result = unsafe { std::mem::zeroed() };
        let query_cstr =
            std::ffi::CString::new(query).map_err(|_| vortex_err!("Invalid query string"))?;

        let status =
            unsafe { cpp::duckdb_query(self.as_ptr(), query_cstr.as_ptr(), &raw mut result) };

        if status != cpp::duckdb_state::DuckDBSuccess {
            let error_msg = unsafe {
                let error_ptr = cpp::duckdb_result_error(&raw mut result);
                if error_ptr.is_null() {
                    "Unknown DuckDB error".to_string()
                } else {
                    CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
                }
            };

            unsafe { cpp::duckdb_destroy_result(&raw mut result) };
            return Err(vortex_err!("Failed to execute query: {}", error_msg));
        }

        Ok(unsafe { QueryResult::new(result) })
    }

    /// Get the client context for this connection.
    pub fn client_context(&self) -> VortexResult<&ClientContextRef> {
        unsafe {
            let client_context = cpp::duckdb_vx_connection_get_client_context(self.as_ptr());
            if client_context.is_null() {
                vortex_bail!(
                    "Failed to get client context: connection={:p}",
                    self.as_ptr()
                )
            }
            Ok(ClientContext::borrow(client_context))
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;

    use super::*;
    use crate::cpp::duckdb_string_t;
    use crate::duckdb::Database;

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
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_query_and_get_row_count_create_table() {
        let conn = test_connection().unwrap();

        // CREATE TABLE should return 0 rows
        let result = conn
            .query("CREATE TABLE test (id INTEGER, name VARCHAR)")
            .unwrap();
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn test_query_and_get_row_count_insert() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER, name VARCHAR)")
            .unwrap();

        let result = conn
            .query("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
            .unwrap();

        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_query_single_value() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 42").unwrap();
        let chunk = result.into_iter().next().unwrap();
        let vec = chunk.get_vector(0);
        let slice = vec.as_slice_with_len::<i32>(chunk.len().as_());

        assert_eq!(chunk.column_count(), 1);
        assert_eq!(chunk.len(), 1);
        assert_eq!(slice[0], 42);
    }

    #[test]
    fn test_query_multiple_rows() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER)").unwrap();
        conn.query("INSERT INTO test VALUES (1), (2), (3)").unwrap();

        let result = conn.query("SELECT id FROM test ORDER BY id").unwrap();
        let chunk = result.into_iter().next().unwrap();
        let vec = chunk.get_vector(0);
        let slice = vec.as_slice_with_len::<i32>(chunk.len().as_());

        assert_eq!(chunk.column_count(), 1);
        assert_eq!(chunk.len(), 3);
        assert_eq!(slice, [1, 2, 3]);
    }

    #[test]
    fn test_query_multiple_columns() {
        let conn = test_connection().unwrap();
        let result = conn.query("SELECT 1 as num, 'hello' as text").unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column_name(0).unwrap(), "num");
        assert_eq!(result.column_name(1).unwrap(), "text");

        let mut chunk = result.into_iter().next().unwrap();
        let len = chunk.len().as_();
        assert_eq!(len, 1);

        let int_val = chunk.get_vector(0).as_slice_with_len::<i32>(len)[0];
        assert_eq!(int_val, 1);

        let vec_str = chunk.get_vector_mut(1);
        let slice_str = unsafe { vec_str.as_slice_mut::<duckdb_string_t>(len) };
        assert_eq!(
            unsafe {
                CStr::from_ptr(cpp::duckdb_string_t_data(&raw mut slice_str[0])).to_string_lossy()
            },
            "hello"
        );
    }

    #[test]
    fn test_query_column_types() {
        let conn = test_connection().unwrap();
        let result = conn
            .query("SELECT 1 as int_col, 'text' as str_col")
            .unwrap();

        assert_eq!(
            result.column_type(0).as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER
        );
        assert_eq!(
            result.column_type(1).as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR
        );
    }

    #[test]
    fn test_null_handling() {
        let conn = test_connection().unwrap();
        let result = conn
            .query("SELECT NULL as null_col, 1 as not_null_col")
            .unwrap();
        let chunk = result.into_iter().next().unwrap();
        let col0 = chunk.get_vector(0);
        let col1 = chunk.get_vector(1);

        assert!(col0.row_is_null(0));
        assert!(!col1.row_is_null(0));
    }

    #[test]
    fn test_type_conversion() {
        let conn = test_connection().unwrap();
        let result = conn
            .query("SELECT 42::TINYINT, 42::SMALLINT, 42::INTEGER, 42::BIGINT")
            .unwrap();
        let chunk = result.into_iter().next().unwrap();
        let vec0 = chunk.get_vector(0);
        let vec1 = chunk.get_vector(1);
        let vec2 = chunk.get_vector(2);
        let vec3 = chunk.get_vector(3);
        let slice0 = vec0.as_slice_with_len::<i8>(chunk.len().as_());
        let slice1 = vec1.as_slice_with_len::<i16>(chunk.len().as_());
        let slice2 = vec2.as_slice_with_len::<i32>(chunk.len().as_());
        let slice3 = vec3.as_slice_with_len::<i64>(chunk.len().as_());

        assert_eq!(slice0[0], 42); // TINYINT -> i64
        assert_eq!(slice1[0], 42); // SMALLINT -> i64
        assert_eq!(slice2[0], 42); // INTEGER -> i64
        assert_eq!(slice3[0], 42); // BIGINT -> i64
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
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_query_and_get_row_count_delete() {
        let conn = test_connection().unwrap();
        conn.query("CREATE TABLE test (id INTEGER)").unwrap();
        conn.query("INSERT INTO test VALUES (1), (2), (3)").unwrap();

        let result = conn.query("DELETE FROM test WHERE id > 1").unwrap();
        assert_eq!(result.row_count(), 2);
    }
}
