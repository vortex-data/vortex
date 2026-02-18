// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::ffi::c_char;
use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::copy::VortexCopyFunction;
pub use crate::duckdb::Connection;
pub use crate::duckdb::Database;
pub use crate::duckdb::LogicalType;
pub use crate::duckdb::Value;
use crate::scan::VortexTableFunction;

pub mod convert;
pub mod duckdb;
pub mod exporter;
mod scan;

#[rustfmt::skip]
#[path = "./cpp.rs"]
/// This module provides the FFI interface to our C++ code exposing additional functionality
/// for DuckDB, such as custom data types and functions.
/// cbindgen:ignore
mod cpp;
mod copy;
#[cfg(test)]
mod e2e_test;

// A global runtime for Vortex operations within DuckDB.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));

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

    database
        .register_vortex_scan_replacement()
        .vortex_expect("failed to register vortex scan replacement");

    let conn = database
        .connect()
        .inspect_err(|e| println!("err {e}"))
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
