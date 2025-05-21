use std::ptr;

use duckdb::core::DataChunkHandle;
use duckdb::ffi::duckdb_data_chunk;
use vortex::ToCanonical;
use vortex::error::VortexExpect;
use vortex_duckdb::{ConversionCache, DuckDBExporter};

use super::{into_conversion_cache, vx_conversion_cache};
use crate::array::vx_array;
use crate::error::{try_or, vx_error};

/// A type for exporting Vortex arrays to a stream of mutable DuckDB vectors.
// TODO(ngates): if this works, we should just wrap up an ArrayIterator and export the whole
//  thing ourselves.
#[allow(non_camel_case_types)]
pub struct vx_duckdb_exporter(DuckDBExporter);

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_create(
    array: *const vx_array,
    error: *mut *mut vx_error,
) -> *mut vx_duckdb_exporter {
    try_or(error, ptr::null_mut(), || {
        let struct_array = unsafe { array.as_ref().vortex_expect("null array") }
            .inner
            .to_struct()?;
        let exporter = DuckDBExporter::try_new(&struct_array)?;
        Ok(Box::into_raw(Box::new(vx_duckdb_exporter(exporter))))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_export(
    exporter: *mut vx_duckdb_exporter,
    data_chunk_ptr: duckdb_data_chunk,
    cache: *mut vx_conversion_cache,
    error: *mut *mut vx_error,
) -> bool {
    let exporter = &mut unsafe { exporter.as_mut().vortex_expect("exporter null") }.0;
    let data_chunk_handle = &mut unsafe { DataChunkHandle::new_unowned(data_chunk_ptr) };
    let cache: &mut ConversionCache = unsafe { into_conversion_cache(cache) };

    try_or(error, false, || exporter.export(data_chunk_handle, cache))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_exporter_free(exporter: *mut vx_duckdb_exporter) {
    assert!(!exporter.is_null());
    drop(unsafe { Box::from_raw(exporter) });
}
