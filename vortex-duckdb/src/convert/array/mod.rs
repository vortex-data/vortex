mod array_ref;
mod cache;
mod data_chunk_adaptor;
mod decimal;
mod table;

pub use cache::ConversionCache;
pub use data_chunk_adaptor::NamedDataChunk;
pub use decimal::precision_to_duckdb_storage_size;
use duckdb::vtab::arrow::WritableVector;
use vortex::ArrayRef;
use vortex::error::VortexResult;

/// Takes an array `self` and a target `chunk` (a duckdb vector), and writes the values from `self`
/// into `chunk`.
/// An `cache` is also provided which can optionally be used to store intermediate expensive
/// to compute values in.
/// The capacity of the vector must be non-strictly larger that the len of the struct array.
pub trait ToDuckDB {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()>;
}

/// Takes a duckdb `vector` and returns a vortex array.
pub trait FromDuckDB<V> {
    fn from_duckdb(vector: V) -> VortexResult<ArrayRef>;
}
