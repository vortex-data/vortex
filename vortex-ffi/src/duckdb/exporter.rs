use std::ptr;

use duckdb::core::DataChunkHandle;
use duckdb::ffi::duckdb_data_chunk;
use vortex::error::VortexExpect;
use vortex_duckdb::ArrayIteratorExporter;

use crate::array::vx_array_iterator;
use crate::error::{try_or, vx_error};

/// A type for exporting Vortex arrays to a stream of mutable DuckDB vectors.
#[allow(non_camel_case_types)]
pub struct vx_duckdb_exporter(ArrayIteratorExporter);

// Create a new array exporter, takes ownership of the array iterator.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_create(
    iter: *mut vx_array_iterator,
    error: *mut *mut vx_error,
) -> *mut vx_duckdb_exporter {
    try_or(error, ptr::null_mut(), || {
        let iter = unsafe { Box::from_raw(iter) }
            .inner
            .vortex_expect("Array iterator already consumed");
        let exporter = ArrayIteratorExporter::new(iter);
        Ok(Box::into_raw(Box::new(vx_duckdb_exporter(exporter))))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_next(
    exporter: *mut vx_duckdb_exporter,
    data_chunk_ptr: duckdb_data_chunk,
    error: *mut *mut vx_error,
) -> bool {
    let exporter = &mut unsafe { exporter.as_mut().vortex_expect("exporter null") }.0;
    let data_chunk_handle = &mut unsafe { DataChunkHandle::new_unowned(data_chunk_ptr) };

    try_or(error, false, || exporter.export(data_chunk_handle))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_free(exporter: *mut vx_duckdb_exporter) {
    assert!(!exporter.is_null());
    drop(unsafe { Box::from_raw(exporter) });
}
