mod cache;

use std::cmp::min;
use std::ffi::{c_char, c_int, c_uchar, c_uint};
use std::ptr;
use std::sync::Arc;

use duckdb::core::{DataChunkHandle, LogicalTypeHandle};
use duckdb::ffi::{duckdb_data_chunk, duckdb_logical_type};
use itertools::Itertools;
use vortex::dtype::{DType, Nullability, StructDType};
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::{Array, ArrayRef, ToCanonical};
use vortex_duckdb::{
    ConversionCache, DUCKDB_STANDARD_VECTOR_SIZE, FromDuckDB, FromDuckDBType, NamedDataChunk,
    ToDuckDBType, to_duckdb_chunk,
};

use crate::array::vx_array;
use crate::duckdb::cache::{into_conversion_cache, vx_conversion_cache};
use crate::error::{try_or, vx_error};
use crate::to_string;

/// Converts a DType into a duckdb
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_to_duckdb_logical_type(
    dtype: *mut DType,
    error: *mut *mut vx_error,
) -> duckdb_logical_type {
    let dtype = unsafe { dtype.as_ref().vortex_expect("null dtype") };

    try_or(error, ptr::null_mut(), || {
        Ok(dtype.to_duckdb_type()?.into_owning_ptr())
    })
}

/// Converts a DuckDB type into a vortex type
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_logical_type_to_dtype(
    column_types: *const duckdb_logical_type,
    column_nullable: *const c_uchar,
    column_names: *const *const c_char,
    column_count: c_int,
    error: *mut *mut vx_error,
) -> *mut DType {
    try_or(error, ptr::null_mut(), || {
        let field_names: Vec<Arc<str>> = (0..column_count)
            .map(|idx| unsafe { to_string(*column_names.offset(idx as isize)) })
            .map(Arc::from)
            .collect();

        let types = (0..column_count)
            .map(|idx| unsafe {
                (
                    LogicalTypeHandle::new_unowned(*column_types.offset(idx as isize)),
                    *column_nullable.offset(idx as isize) != 0,
                )
            })
            .map(|(type_, nullable)| DType::from_duckdb(type_, nullable.into()))
            .collect::<VortexResult<Vec<DType>>>()?;

        // Top level structs cannot be nullable sql/duckdb.
        let dtype = Box::new(DType::Struct(
            Arc::new(StructDType::new(field_names.into(), types)),
            Nullability::NonNullable,
        ));

        Ok(Box::into_raw(dtype))
    })
}

/// Back a single chunk of the array as a duckdb data chunk.
/// The initial call should pass offset = 0.
/// The offset is returned to the caller, which can be used to request the next chunk.
/// 0 is returned when the stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_to_duckdb_chunk(
    stream: *mut vx_array,
    offset: c_uint,
    data_chunk_ptr: duckdb_data_chunk,
    cache: *mut vx_conversion_cache,
    error: *mut *mut vx_error,
) -> c_uint {
    try_or(error, 0, || {
        let offset = offset as usize;

        let array = &unsafe { stream.as_ref() }
            .vortex_expect("null stream")
            .inner;

        assert!(array.len() > offset, "offset out of bounds");

        let end = min(offset + DUCKDB_STANDARD_VECTOR_SIZE, array.len());
        let is_end = end == array.len();

        let slice = array.slice(offset, end)?;
        let mut data_chunk_handle = unsafe { DataChunkHandle::new_unowned(data_chunk_ptr) };
        let cache: &mut ConversionCache = unsafe { into_conversion_cache(cache) };

        to_duckdb_chunk(
            &slice.to_struct().vortex_expect("must be a struct"),
            &mut data_chunk_handle,
            cache,
        )?;

        if is_end {
            Ok(0)
        } else {
            Ok(u32::try_from(end)?)
        }
    })
}

/// Pushed a single duckdb chunk into a file sink.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_duckdb_chunk_to_array(
    chunk: duckdb_data_chunk,
    dtype: *mut DType,
    error: *mut *mut vx_error,
) -> *mut vx_array {
    let dtype = unsafe { dtype.as_ref().vortex_expect("null array") };
    try_or(error, ptr::null_mut(), || {
        let struct_type = dtype.as_struct().ok_or_else(|| {
            vortex_err!("cannot push a duckdb to an array stream which is not a top level struct")
        })?;

        let nullable = struct_type
            .fields()
            .map(|f| f.nullability() == Nullability::Nullable)
            .collect_vec();

        let array = ArrayRef::from_duckdb(&NamedDataChunk {
            chunk: &unsafe { DataChunkHandle::new_unowned(chunk) },
            nullable: Some(&nullable),
            names: Some(struct_type.names().clone()),
        })?;

        Ok(Box::into_raw(Box::new(vx_array { inner: array })))
    })
}

#[cfg(test)]
mod tests {
    use std::ptr::null_mut;

    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::arrays::{PrimitiveArray, StructArray};
    use vortex::error::VortexExpect;

    use crate::array::vx_array;
    use crate::duckdb::cache::{vx_conversion_cache_create, vx_conversion_cache_free};
    use crate::duckdb::vx_array_to_duckdb_chunk;
    use crate::error::vx_error;

    #[test]
    fn test_long_array() {
        let vortex: PrimitiveArray = (0i32..4095).collect();
        let vortex = StructArray::from_fields(&[("a", vortex.to_array())]).vortex_expect("str");

        let ffi_array: *mut vx_array = Box::into_raw(Box::new(vx_array {
            inner: vortex.to_array(),
        }));

        let cache = unsafe { vx_conversion_cache_create(0) };

        let mut error: *mut vx_error = null_mut();

        let handle = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        let offset =
            unsafe { vx_array_to_duckdb_chunk(ffi_array, 0, handle.get_ptr(), cache, &mut error) };
        assert!(error.is_null());
        assert_eq!(offset, 2048);
        assert_eq!(handle.len(), 2048);

        let offset = unsafe {
            vx_array_to_duckdb_chunk(ffi_array, offset, handle.get_ptr(), cache, &mut error)
        };
        assert!(error.is_null());
        assert_eq!(offset, 0);
        assert_eq!(handle.len(), 2047);

        unsafe {
            vx_conversion_cache_free(cache);
        }
    }
}
