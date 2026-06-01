// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::ffi::c_char;
use std::sync::LazyLock;
use std::sync::OnceLock;

use vortex::VortexSessionDefault;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::duckdb::Database;
use crate::duckdb::DatabaseRef;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;

mod column_statistics;
mod convert;
pub mod duckdb;
mod exporter;
mod ffi;
mod filesystem;
mod multi_file;
mod projection;
mod table_function;

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
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default().with_handle(RUNTIME.handle());
    vortex_geo::initialize(&session);
    session
});

// Duckdb's logger requires a *Context as first argument which
// would be hard to integrate with tracing::. We use logging for
// debugging only anyway, so that's good enough.
fn init_tracing() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        drop(
            tracing_subscriber::fmt()
                .with_writer(std::io::stdout)
                .try_init(),
        );
    });
}

/// Initialize the Vortex extension by registering the extension functions.
/// Note: This also registers extension options. If you want to register options
/// separately (e.g., before creating connections), call `register_extension_options` first.
pub fn initialize(db: &DatabaseRef) -> VortexResult<()> {
    db.config().add_extension_options(
        "vortex_filesystem",
        "Whether to use Vortex's filesystem ('vortex') or DuckDB's filesystems ('duckdb').",
        LogicalType::varchar(),
        Value::from("vortex"),
    )?;
    db.register_table_functions()?;
    db.register_copy_function()
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
    init_tracing();
    let database = unsafe { Database::borrow(db) };

    database
        .register_vortex_scan_replacement()
        .vortex_expect("failed to register vortex scan replacement");
    initialize(database).vortex_expect("Failed to initialize Vortex extension");
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
