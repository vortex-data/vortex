// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::missing_safety_doc)]

use std::ffi::c_char;
use std::ffi::c_void;

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
pub unsafe extern "C" fn vortex_init_rust(db: *mut c_void) {
    unsafe { vortex_duckdb::initialize_extension_from_raw(db) };
}

/// The DuckDB extension ABI version function.
/// This function returns the version of the DuckDB library the extension is built against.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_version_rust() -> *const c_char {
    vortex_duckdb::duckdb_library_version()
}

/// An additional function we export to expose the version of the extension itself to C++ code.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_extension_version_rust() -> *const c_char {
    vortex_duckdb::extension_version()
}
