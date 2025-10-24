// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::missing_safety_doc)]
use std::ffi::{CStr, c_char};

// **WARNING end
use vortex::error::{VortexExpect, VortexResult};

use crate::copy::VortexCopyFunction;
use crate::duckdb::Config;
pub use crate::duckdb::{Connection, Database, LogicalType, Value};
use crate::scan::VortexTableFunction;

mod convert;
pub mod duckdb;
pub mod exporter;
mod scan;
mod utils;

#[rustfmt::skip]
#[path = "./cpp.rs"]
/// This module provides the FFI interface to our C++ code exposing additional functionality
/// for DuckDB, such as custom data types and functions.
/// cbindgen:ignore
mod cpp;
mod copy;
#[cfg(test)]
mod e2e_test;

/// Register Vortex extension configuration options with DuckDB.
/// This must be called before `register_table_functions` to take effect.
pub fn register_extension_options(config: &Config) {
    let logical_type = LogicalType::uint64();

    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1);
    let default_value = Value::from(default_threads);

    // Register the vortex_max_threads extension option
    // SAFETY: We're passing valid pointers for database, logical_type, and default_value
    // The C++ code will copy the LogicalType and Value, so we can safely drop them after this call
    let result = unsafe {
        cpp::duckdb_vx_add_extension_option(
            config.as_ptr(),
            c"vortex_max_threads".as_ptr(),
            c"Maximum number of threads for Vortex table scans".as_ptr(),
            logical_type.as_ptr(),
            default_value.as_ptr(),
        )
    };

    assert_eq!(
        result,
        cpp::duckdb_state::DuckDBSuccess,
        "Failed to register vortex_max_threads extension option"
    );
}

/// Initialize the Vortex extension by registering the extension functions.
/// Note: This also registers extension options. If you want to register options
/// separately (e.g., before creating connections), call `register_extension_options` first.
pub fn register_table_functions(conn: &Connection) -> VortexResult<()> {
    conn.register_table_function::<VortexTableFunction>(c"vortex_scan")?;
    conn.register_table_function::<VortexTableFunction>(c"read_vortex")?;
    conn.register_copy_function::<VortexCopyFunction>(c"vortex", c"vortex")
}

/// Global symbol visibility in the Vortex extension:
/// - Rust functions use C ABI with "_rust" suffix (e.g., vortex_init_rust)
/// - C++ wrapper functions have the expected name without suffix (e.g., vortex_init)
/// - C++ wrappers are annotated with DUCKDB_EXTENSION_API to ensure global visibility
/// - C++ wrappers call the corresponding Rust functions
///
/// This ensures DuckDB can find the symbols when loading the extension.
///
/// The DuckDB extension ABI initialization function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_init_rust(db: cpp::duckdb_database) {
    let database = unsafe { Database::borrow(db) };

    let conn = database
        .connect()
        .vortex_expect("Failed to connect to DuckDB database");
    register_table_functions(&conn).vortex_expect("Failed to initialize Vortex extension");
}

/// The DuckDB extension ABI version function.
/// This function returns the version of the DuckDB library the extension is built against.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_version_rust() -> *const c_char {
    unsafe { cpp::duckdb_library_version() }
}

/// An additional function we export to expose the version of the extension itself to C++ code.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_extension_version_rust() -> *const c_char {
    // We do some fiddly macros here to get ourselves a _static_ C-style string.
    // Otherwise, we'd be leaking memory.
    unsafe {
        CStr::from_bytes_with_nul_unchecked(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes())
    }
    .as_ptr()
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;
    use crate::duckdb::{Config, Database};

    #[test]
    fn test_vortex_max_threads_option_registration() {
        let config = Config::new().expect("Failed to create config");
        register_extension_options(&config);
        let db = Database::open_in_memory_with_config(config).expect("Failed to open database");

        let conn = db.connect().expect("Failed to connect");

        let _result1 = conn
            .query("SET vortex_max_threads = 4")
            .expect("Failed to set vortex_max_threads - option may not be registered");

        let max_threads_cstr = CString::new("vortex_max_threads").unwrap();
        let ctx = conn.client_context().vortex_expect("ctx exists");
        assert_eq!(
            ctx.try_get_current_setting(&max_threads_cstr)
                .unwrap()
                .to_string(),
            "4"
        );

        let _result2 = conn
            .query("SET vortex_max_threads = 8")
            .expect("Failed to set vortex_max_threads to 8");

        assert_eq!(
            ctx.try_get_current_setting(&max_threads_cstr)
                .unwrap()
                .to_string(),
            "8"
        );
    }
}
