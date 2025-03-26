#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::missing_safety_doc)]

/// This is the default chunk size for duckdb.
/// It is best to return data chunks of this size to duckdb.
pub const DUCKDB_STANDARD_VECTOR_SIZE: usize = 2048;

mod convert;

use std::ffi::c_char;

pub use convert::{FromDuckDB, FromDuckDBType, ToDuckDB, ToDuckDBType, to_duckdb_chunk};

// To generate C decls to include in vortex_duckdb_extension.cpp,
// call `cbindgen` from `vortex/vortex-duckdb`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_duckdb_hello() -> *const c_char {
    c"Hello, world from Rust! ".as_ptr()
}
