use duckdb::core::DataChunkHandle;
use duckdb::ffi::duckdb_data_chunk;
use vortex_duckdb::ArrayIteratorExporter;

use crate::array_iterator::vx_array_iterator;
use crate::box_wrapper;
use crate::error::{try_or, vx_error};

box_wrapper!(
    /// A type for exporting Vortex arrays to a stream of mutable DuckDB vectors.
    ArrayIteratorExporter,
    vx_duckdb_exporter
);

// Create a new array exporter, taking ownership of the array iterator.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_new(
    iter: *mut vx_array_iterator,
) -> *mut vx_duckdb_exporter {
    vx_duckdb_exporter::new(Box::new(ArrayIteratorExporter::new(
        vx_array_iterator::into_box(iter),
    )))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_next(
    exporter: *mut vx_duckdb_exporter,
    data_chunk_ptr: duckdb_data_chunk,
    error: *mut *const vx_error,
) -> bool {
    let exporter = vx_duckdb_exporter::as_mut(exporter);
    let data_chunk_handle = &mut unsafe { DataChunkHandle::new_unowned(data_chunk_ptr) };

    try_or(error, false, || exporter.export(data_chunk_handle))
}
