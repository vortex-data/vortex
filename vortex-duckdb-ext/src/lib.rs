#![allow(clippy::missing_safety_doc)]
use std::ffi::{CStr, c_char};

use vortex::error::{VortexExpect, VortexResult};

use crate::duckdb::{Connection, Database};
use crate::scan::HelloTableFunction;

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

/// Initialize the Vortex extension by registering the `hello` function.
pub fn init(conn: &Connection) -> VortexResult<()> {
    conn.register_table_function::<HelloTableFunction>(c"hello")
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
    // TODO(alex): bring back tests
    // use duckdb::Connection;

    // use crate::duckdb::Database;

    #[test]
    fn test_extension() {
        // let db = Database::open_in_memory().unwrap();
        // let connection = db.connect().unwrap();
        // super::init(&connection).unwrap();

        // // Now we use DuckDB-rs to query the connection.
        // let conn = unsafe { Connection::open_from_raw(db.as_ptr().cast()) }.unwrap();
        // let result = conn
        //     .prepare("SELECT * FROM hello(?) WHERE greeting = 'Hello Bob'")
        //     .unwrap()
        //     .query_row(["Bob"], |row| row.get::<_, String>(0))
        //     .unwrap();
        // assert_eq!(&result, "Hello Bob");
    }
}
