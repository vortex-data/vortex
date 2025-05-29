mod exporter;

use std::ffi::{c_char, c_int, c_uchar};
use std::ptr;
use std::sync::Arc;

use duckdb::core::{DataChunkHandle, LogicalTypeHandle};
use duckdb::ffi::{duckdb_data_chunk, duckdb_logical_type};
use itertools::Itertools;
use vortex::ArrayRef;
use vortex::dtype::{DType, Nullability, StructDType};
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex_duckdb::{FromDuckDB, FromDuckDBType, NamedDataChunk, ToDuckDBType};

use crate::array::vx_array;
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
