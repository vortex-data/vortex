use std::cmp::min;
use std::ffi::c_uint;

use duckdb::core::DataChunkHandle;
use duckdb::ffi::{duckdb_data_chunk, duckdb_logical_type};
use vortex::compute::slice;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::{Array, ToCanonical};
use vortex_duckdb::{DUCKDB_STANDARD_VECTOR_SIZE, ToDuckDBType, to_duckdb_chunk};

use crate::array::FFIArray;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_to_duckdb_logical_type(dtype: *mut DType) -> duckdb_logical_type {
    let dtype = unsafe { &*dtype };

    dtype
        .to_duckdb_type()
        .vortex_expect("convert to duckdb")
        .into_owning_ptr()
}

/// Back a single chunk of the array as a duckdb data chunk.
/// The initial call should pass offset = 0.
/// The offset is returned to the caller, which can be used to request the next chunk.
/// 0 is returned when the stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArray_to_duckdb_chunk(
    stream: *mut FFIArray,
    offset: c_uint,
    data_chunk_ptr: duckdb_data_chunk,
) -> c_uint {
    let offset = offset as usize;
    let array = unsafe { &(*stream).inner };

    assert!(array.len() > offset, "offset out of bounds");

    let end = min(offset + DUCKDB_STANDARD_VECTOR_SIZE, array.len());
    let is_end = end == array.len();

    let slice = slice(array, offset, end).vortex_expect("slice");
    let mut data_chunk_handle = unsafe { DataChunkHandle::new_unowned(data_chunk_ptr) };
    to_duckdb_chunk(
        &slice.to_struct().vortex_expect("must be a struct"),
        &mut data_chunk_handle,
    )
    .vortex_expect("to_duckdb");

    if is_end {
        0
    } else {
        u32::try_from(end).vortex_expect("end overruns u32")
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::Array;
    use vortex::arrays::{PrimitiveArray, StructArray};
    use vortex::error::VortexExpect;

    use crate::array::FFIArray;
    use crate::duckdb::FFIArray_to_duckdb_chunk;

    #[test]
    fn test_long_array() {
        let vortex: PrimitiveArray = (0i32..4095).collect();
        let vortex = StructArray::from_fields(&[("a", vortex.to_array())]).vortex_expect("str");

        let ffi_array: *mut FFIArray = Box::into_raw(Box::new(FFIArray {
            inner: vortex.to_array(),
        }));

        let handle = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        let offset = unsafe { FFIArray_to_duckdb_chunk(ffi_array, 0, handle.get_ptr()) };
        assert_eq!(offset, 2048);
        assert_eq!(handle.len(), 2048);
        let offset = unsafe { FFIArray_to_duckdb_chunk(ffi_array, offset, handle.get_ptr()) };
        assert_eq!(offset, 0);
        assert_eq!(handle.len(), 2047);
    }
}
