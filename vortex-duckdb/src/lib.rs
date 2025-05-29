#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::missing_safety_doc)]

/// This is the default chunk size for duckdb.
/// It is best to return data chunks of this size to duckdb.
/// 2048 is the default chunk size for duckdb.
pub const DUCKDB_STANDARD_VECTOR_SIZE: usize = 2048;

mod buffer;
mod convert;
mod exporter;

pub use convert::*;
pub use exporter::*;

// Note: To generate C decls to include in vortex_duckdb_extension.cpp,
// call `cbindgen` from `vortex/vortex-duckdb`.

#[cfg(test)]
mod tests {
    use duckdb::ffi::duckdb_vector_size;

    use crate::DUCKDB_STANDARD_VECTOR_SIZE;

    #[test]
    fn assert_duckdb_vector_size_matches() {
        assert_eq!(
            Ok(DUCKDB_STANDARD_VECTOR_SIZE),
            usize::try_from(unsafe { duckdb_vector_size() })
        );
    }
}
