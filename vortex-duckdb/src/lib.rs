// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::missing_safety_doc)]

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
use crate::duckdb::Database;
use crate::duckdb::DatabaseRef;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;
use crate::multi_file::VortexMultiFileScan;
use crate::multi_file::VortexMultiFileScanList;
use crate::multi_file_function::VortexMultiFileFunction;

mod convert;
mod datasource;
pub mod duckdb;
mod exporter;
mod filesystem;
mod multi_file;
mod multi_file_function;

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

/// Returns true if the user has opted into the experimental MultiFileFunction-
/// backed scan path via `VX_DUCKDB_MULTI_FILE_FUNCTION=1` (or `=true`).
///
/// Used to switch between the existing TableFunction-driven `read_vortex` and
/// the new `MultiFileFunction<OP>`-driven path during benchmarking. Defaults
/// to off so the existing scan remains the path of record.
///
/// Known gaps in the v2 path (compared to v1) at time of writing:
/// - No batch parallelism within a file (`TryInitializeScan` is one-shot, so
///   each Vortex file is scanned by a single worker).
/// - No `union_by_name`, hive partitioning columns, or `filename` /
///   `file_row_number` virtual columns wired through.
/// - No support for the named parameters DuckDB's `MultiFileReader` adds
///   (`union_by_name`, `hive_partitioning`, …) — `ParseOption` returns false.
/// - No `COPY ... FROM 'x.vortex'` via this path.
///
/// These are tracked as follow-up work; for now `read_vortex_v2` exists
/// alongside `read_vortex` so orchestration paths can be benchmarked
/// side-by-side.
fn use_multi_file_function() -> bool {
    matches!(
        std::env::var("VX_DUCKDB_MULTI_FILE_FUNCTION").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
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
    if use_multi_file_function() {
        // Replace the table-function-based scan with the MultiFileFunction<OP>
        // path under the canonical names. Also expose under v2 names so an A/B
        // test can run both registrations side-by-side.
        db.register_multi_file_function::<VortexMultiFileFunction>(c"vortex_scan")?;
        db.register_multi_file_function::<VortexMultiFileFunction>(c"read_vortex")?;
    } else {
        db.register_table_function::<VortexMultiFileScan>(c"vortex_scan")?;
        db.register_table_function::<VortexMultiFileScan>(c"read_vortex")?;
        // Register list overloads for multi-glob scanning (e.g., read_vortex(['a.vortex', 'b.vortex']))
        db.register_table_function::<VortexMultiFileScanList>(c"vortex_scan")?;
        db.register_table_function::<VortexMultiFileScanList>(c"read_vortex")?;
    }
    // Always expose the v2 path under its own name so it can be invoked
    // explicitly without flipping the env var (useful for A/B testing within
    // a single process).
    db.register_multi_file_function::<VortexMultiFileFunction>(c"read_vortex_v2")?;
    db.register_copy_function::<VortexCopyFunction>(c"vortex", c"vortex")
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
