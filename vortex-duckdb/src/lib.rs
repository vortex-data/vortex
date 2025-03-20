#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::missing_safety_doc)]

mod convert;

use std::ffi::c_char;

pub use convert::{FromDuckDB, FromDuckDBType, ToDuckDB, ToDuckDBType, to_duckdb_chunk};

// To generate C decls to include in vortex_duckdb_extension.cpp,
// call `cbindgen` from `vortex/vortex-duckdb`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_duckdb_hello() -> *const c_char {
    c"Hello, world from Rust! ".as_ptr()
}
