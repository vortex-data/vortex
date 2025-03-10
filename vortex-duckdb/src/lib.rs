#![cfg(not(target_arch = "wasm32"))]

mod convert;

pub use convert::{FromDuckDB, FromDuckDBType, ToDuckDB, ToDuckDBType, to_duckdb_chunk};
