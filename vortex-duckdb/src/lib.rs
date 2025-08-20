// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::missing_safety_doc)]
use std::ffi::{CStr, c_char};
use std::sync::LazyLock;

// **WARNING begin this includes duckdb-rs, which is required to link in the symbol from libduckdb-sys.
use tokio::runtime;
use tokio::runtime::Runtime;
// **WARNING end
use vortex::error::{VortexExpect, VortexResult};

use crate::copy::VortexCopyFunction;
pub use crate::duckdb::{Connection, Database};
use crate::scan::VortexTableFunction;

mod convert;
pub mod duckdb;
pub mod exporter;
pub mod optimizer;
mod optimizer_plan;
mod rust_optimizer;
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

/// Initialize the Vortex extension by registering the extension functions.
pub fn register_table_functions(conn: &Connection) -> VortexResult<()> {
    conn.register_table_function::<VortexTableFunction>(c"vortex_scan")?;
    conn.register_table_function::<VortexTableFunction>(c"read_vortex")?;
    conn.register_copy_function::<VortexCopyFunction>(c"vortex", c"vortex")
}

/// Initialize the Vortex extension (table functions AND optimizer)
pub fn register_extension(db: &mut Database) -> VortexResult<()> {
    println!("🚀 REGISTERING: Starting Vortex extension registration...");

    // Register the new Rust-based optimizer first
    println!("🚀 REGISTERING: Registering Rust optimizer...");
    if let Err(e) = optimizer::register_rust_optimizer(db) {
        println!(
            "⚠️ REGISTERING: Rust optimizer registration failed: {}, trying legacy C++ optimizer...",
            e
        );

        // Fallback to legacy C++ optimizer
        if let Err(e2) = optimizer::register_optimizer(db) {
            println!(
                "⚠️ REGISTERING: Both optimizers failed. Rust: {}, C++: {}. Continuing with table functions only.",
                e, e2
            );
        } else {
            println!("✅ REGISTERING: Legacy C++ optimizer registration succeeded!");
        }
    } else {
        println!("✅ REGISTERING: Rust optimizer registration succeeded!");
    }

    // Register table functions
    println!("🚀 REGISTERING: Registering table functions...");
    let conn = db.connect()?;
    let result = register_table_functions(&conn);
    println!("✅ REGISTERING: Extension registration completed!");

    result
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
    println!("🚀 INIT: vortex_init_rust called - registering at DuckDB extension loading time");

    let mut database = unsafe { Database::borrow(db) };

    // Try registering optimizer first, during extension loading
    println!("🚀 INIT: Registering optimizer during extension loading...");
    if let Err(e) = optimizer::register_optimizer(&mut database) {
        println!(
            "⚠️ INIT: Optimizer registration failed: {}, continuing without optimizer",
            e
        );
    } else {
        println!("✅ INIT: Optimizer registration succeeded during extension loading");
    }

    // Register table functions
    println!("🚀 INIT: Registering table functions...");
    let conn = database
        .connect()
        .vortex_expect("Failed to connect to database");
    register_table_functions(&conn).vortex_expect("Failed to register table functions");
    println!("✅ INIT: Extension initialization complete");
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

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Cannot start runtime")
});
