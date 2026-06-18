// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::missing_safety_doc)]

use std::ffi::c_char;
use std::ffi::c_void;
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
#[allow(rustdoc::all)]
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
    db.register_optimizer_extension()?;
    db.register_copy_function()
}

/// Initialize the DuckDB extension from a raw DuckDB database pointer.
pub unsafe fn initialize_extension_from_raw(db: *mut c_void) {
    init_tracing();
    let database = unsafe { Database::borrow(db.cast()) };

    database
        .register_vortex_scan_replacement()
        .vortex_expect("failed to register vortex scan replacement");
    initialize(database).vortex_expect("Failed to initialize Vortex extension");
}

/// Returns the version of the DuckDB library the extension is built against.
pub fn duckdb_library_version() -> *const c_char {
    unsafe { cpp::duckdb_library_version() }
}

/// Returns the version of the Vortex DuckDB extension.
pub fn extension_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}
