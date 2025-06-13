#![allow(clippy::missing_safety_doc)]
use std::ffi::{CStr, c_char};

use vortex::error::{VortexExpect, VortexResult};

use crate::duckdb::{Connection, Database};
use crate::scan::VortexTableFunction;

mod convert;
pub mod duckdb;
pub mod exporter;
mod scan;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
#[allow(clippy::suspicious_doc_comments)]
#[allow(clippy::enum_variant_names)]
#[rustfmt::skip]
#[path = "./cpp.rs"]
/// This module provides the FFI interface to our C++ code exposing additional functionality
/// for DuckDB, such as custom data types and functions.
/// cbindgen:ignore
mod cpp;

/// Initialize the Vortex extension by registering the `vortex_scan` function.
pub fn init(conn: &Connection) -> VortexResult<()> {
    conn.register_table_function::<VortexTableFunction>(c"vortex_scan")
}

/// The DuckDB extension ABI initialization function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_init(db: cpp::duckdb_database) {
    let conn = unsafe { Database::borrow(db) }
        .connect()
        .vortex_expect("Failed to connect to DuckDB database");
    init(&conn).vortex_expect("Failed to initialize Vortex extension");
}

/// The DuckDB extension ABI version function.
/// This function returns the version of the DuckDB library the extension is built against.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_version() -> *const c_char {
    unsafe { cpp::duckdb_library_version() }
}

/// An additional function we export to expose the version of the extension itself to C++ code.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_extension_version() -> *const c_char {
    // We do some fiddly macros here to get ourselves a _static_ C-style string.
    // Otherwise, we'd be leaking memory.
    unsafe {
        CStr::from_bytes_with_nul_unchecked(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes())
    }
    .as_ptr()
}

#[cfg(test)]
mod tests {
    use duckdb::Connection;
    use tempfile::NamedTempFile;
    use vortex::IntoArray;
    use vortex::arrays::{BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray};
    use vortex::file::VortexWriteOptions;
    use vortex::scalar::Scalar;
    use vortex::validity::Validity;

    use crate::duckdb::Database;

    fn database_connection() -> Connection {
        let db = Database::open_in_memory().unwrap();
        let connection = db.connect().unwrap();
        super::init(&connection).unwrap();
        unsafe { Connection::open_from_raw(db.as_ptr().cast()) }.unwrap()
    }

    fn create_temp_file() -> NamedTempFile {
        NamedTempFile::new().unwrap()
    }

    async fn write_vortex_file(field_name: &str, array: impl IntoArray) -> NamedTempFile {
        let temp_file_path = create_temp_file();

        let struct_array = StructArray::from_fields(&[(field_name, array.into_array())]).unwrap();
        let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
        VortexWriteOptions::default()
            .write(file, struct_array.to_array_stream())
            .await
            .unwrap();

        temp_file_path
    }

    fn scan_vortex_file<T>(tmp_file: NamedTempFile, query: &str) -> T
    where
        T: duckdb::types::FromSql,
    {
        let conn = database_connection();
        conn.prepare(query)
            .unwrap()
            .query_row([tmp_file.path().to_string_lossy()], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn test_scan_function_registration() {
        let conn = database_connection();
        let result: String = conn
            .prepare(
                "SELECT function_name FROM duckdb_functions() WHERE function_name = 'vortex_scan'",
            )
            .unwrap()
            .query_row([], |row| row.get(0))
            .unwrap();
        assert_eq!(&result, "vortex_scan");
    }

    #[tokio::test]
    async fn test_vortex_scan_strings() {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        let file = write_vortex_file("strings", strings).await;
        let result: String =
            scan_vortex_file(file, "SELECT string_agg(strings, ',') FROM vortex_scan(?)");
        assert_eq!(result, "Hello,Hi,Hey");
    }

    #[tokio::test]
    async fn test_vortex_scan_integers() {
        let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
        let file = write_vortex_file("number", numbers).await;
        let sum: i64 = scan_vortex_file(file, "SELECT SUM(number) FROM vortex_scan(?)");
        assert_eq!(sum, 138);
    }

    #[tokio::test]
    async fn test_vortex_scan_floats() {
        let values = PrimitiveArray::from_iter([1.5f64, -2.5, 0.0, 42.42]);
        let file = write_vortex_file("value", values).await;
        let count: i64 =
            scan_vortex_file(file, "SELECT COUNT(*) FROM vortex_scan(?) WHERE value > 0");
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_vortex_scan_constant() {
        let constant = ConstantArray::new(Scalar::from(42i32), 100);
        let file = write_vortex_file("constant", constant).await;
        let value: i32 = scan_vortex_file(file, "SELECT constant FROM vortex_scan(?) LIMIT 1");
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn test_vortex_scan_booleans() {
        let flags = vec![true, false, true, true, false];
        let flags_array = BoolArray::new(flags.into(), Validity::NonNullable);
        let file = write_vortex_file("flag", flags_array).await;
        let true_count: i64 = scan_vortex_file(
            file,
            "SELECT COUNT(*) FROM vortex_scan(?) WHERE flag = true",
        );
        assert_eq!(true_count, 3);
    }
}
