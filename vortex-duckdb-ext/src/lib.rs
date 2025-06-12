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
    use std::env;
    use std::path::{Path, PathBuf};

    use duckdb::Connection;
    use vortex::IntoArray;
    use vortex::arrays::{StructArray, VarBinArray};
    use vortex::file::VortexWriteOptions;

    use crate::duckdb::Database;

    fn database_connection() -> Connection {
        let db = Database::open_in_memory().unwrap();
        let connection = db.connect().unwrap();
        super::init(&connection).unwrap();
        unsafe { Connection::open_from_raw(db.as_ptr().cast()) }.unwrap()
    }

    fn temp_file_path() -> PathBuf {
        let temp_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("vortex-duckdb-ext");
        std::fs::create_dir_all(&temp_dir).unwrap();
        temp_dir.join("test-data.vortex")
    }

    #[test]
    fn test_scan_function_registration() {
        let conn = database_connection();

        let result = conn
            .prepare(
                "SELECT function_name FROM duckdb_functions() WHERE function_name = 'vortex_scan'",
            )
            .unwrap()
            .query_row([], |row| row.get::<_, String>(0))
            .unwrap();

        assert_eq!(&result, "vortex_scan");
    }

    #[tokio::test]
    async fn test_vortex_scan_with_file() {
        let temp_file_path = temp_file_path();
        if temp_file_path.exists() {
            // Clear previous temp file.
            std::fs::remove_file(&temp_file_path).unwrap();
        }

        let greetings = ["Hello", "Hi", "Hey"];
        let greetings_array = VarBinArray::from(greetings.to_vec());
        let struct_array =
            StructArray::from_fields(&[("greeting", greetings_array.into_array())]).unwrap();

        // Write test data to Vortex file.
        let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
        VortexWriteOptions::default()
            .write(file, struct_array.to_array_stream())
            .await
            .unwrap();

        let conn = database_connection();

        // Execute the query and run the scan.
        for greeting in greetings.iter() {
            let result: String = conn
                .prepare(&format!(
                    "SELECT greeting FROM vortex_scan(?) WHERE greeting = '{greeting}'"
                ))
                .unwrap()
                .query_row([temp_file_path.to_string_lossy().as_ref()], |row| {
                    // From the row, get the column at index 0.
                    row.get(0)
                })
                .unwrap();

            assert_eq!(&result, greeting);
        }
    }
}
