mod cache;

use std::cmp::min;
use std::ffi::{c_char, c_int, c_uchar, c_uint};
use std::ptr;
use std::sync::Arc;

use duckdb::core::{DataChunkHandle, LogicalTypeHandle};
use duckdb::ffi::{duckdb_data_chunk, duckdb_logical_type};
use itertools::Itertools;
use vortex::arrays::ChunkedArray;
use vortex::dtype::{DType, Nullability, StructDType};
use vortex::error::{VortexExpect, VortexResult};
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

/// Returns an empty vortex array constructed from three arrays of len `len`, the (types, null, names).
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_create_empty_from_duckdb_table(
    type_array: *const duckdb_logical_type,
    nullable: *const c_uchar,
    names: *const *const c_char,
    len: c_int,
    error: *mut *mut vx_error,
) -> *mut vx_array {
    try_or(error, ptr::null_mut(), || {
        let field_names: Vec<Arc<str>> = (0..len)
            .map(|i| to_string(*names.offset(i as isize)))
            .map(Arc::from)
            .collect();

        let types = (0..len)
            .map(|i| {
                (
                    LogicalTypeHandle::new_unowned(unsafe { *type_array.offset(i as isize) }),
                    *nullable.offset(i as isize) != 0,
                )
            })
            .map(|(type_, nullable)| DType::from_duckdb(type_, nullable.into()))
            .collect::<VortexResult<Vec<DType>>>()?;

        let file_dtype = DType::Struct(
            Arc::new(StructDType::new(field_names.into(), types)),
            Nullability::NonNullable,
        );

        let chunked_array = ChunkedArray::try_new(vec![], file_dtype).vortex_expect("cannot fail");

        let ffi_array = vx_array {
            inner: chunked_array.to_array(),
        };

        Ok(Box::leak(Box::new(ffi_array)))
    })
}

/// Requires a vortex array, a duckdb data chunk and a nullable array (equal to len(chunk.columns)).
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_append_duckdb_chunk(
    array: *mut vx_array,
    chunk: duckdb_data_chunk,
    nullable: *const c_uchar,
) -> *mut vx_array {
    let array = unsafe { array.as_ref().vortex_expect("null array") };

    let struct_type = array
        .inner
        .dtype()
        .as_struct()
        .vortex_expect("can only write a struct array from duckdb");

    let chunked_array = array
        .inner
        .as_any()
        .downcast_ref::<ChunkedArray>()
        .vortex_expect("can only append to chunked array");

    let chunk = DataChunkHandle::new_unowned(chunk);

    let nullable = (0..chunk.num_columns())
        .map(|i| *nullable.add(i) != 0)
        .collect_vec();

    let new_chunk = ArrayRef::from_duckdb(&NamedDataChunk {
        chunk: &chunk,
        nullable: Some(&nullable),
        names: Some(struct_type.names().clone()),
    })
    .vortex_expect("from_duckdb convert");

    let mut chunks = chunked_array.chunks().to_vec();
    chunks.push(new_chunk);

    let chunked_array = ChunkedArray::try_new(chunks, chunked_array.dtype().clone())
        .vortex_expect("appending array");

    Box::leak(Box::new(vx_array {
        inner: chunked_array.to_array(),
    })) as *mut vx_array
}

#[cfg(test)]
mod tests {
    use std::ptr::null_mut;

    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::Array;
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
