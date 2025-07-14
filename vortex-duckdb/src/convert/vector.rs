// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Copied of duckdb-rs (https://github.com/duckdb/duckdb-rs/blob/main/crates/duckdb/src/vtab/arrow.rs)
use std::sync::Arc;

use arrow_array::builder::GenericBinaryBuilder;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, Time64MicrosecondType,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, BooleanArray, Date32Array, PrimitiveArray, StringArray, TimestampMicrosecondArray,
    TimestampNanosecondArray,
};
use arrow_buffer::buffer::{BooleanBuffer, NullBuffer};
use bitvec::macros::internal::funty::Fundamental;
use vortex::ArrayRef;
use vortex::arrays::StructArray;
use vortex::arrow::FromArrowArray;
use vortex::dtype::FieldNames;
use vortex::error::{VortexResult, vortex_err};

use crate::cpp::{
    DUCKDB_TYPE, duckdb_date, duckdb_string_t, duckdb_string_t_data, duckdb_string_t_length,
    duckdb_time, duckdb_timestamp,
};
use crate::duckdb::{DataChunk, Vector};

pub struct DuckString<'a> {
    ptr: &'a mut duckdb_string_t,
}

impl<'a> DuckString<'a> {
    pub(crate) fn new(ptr: &'a mut duckdb_string_t) -> Self {
        DuckString { ptr }
    }
}

impl<'a> DuckString<'a> {
    /// convert duckdb_string_t to a copy on write string
    pub fn as_str(&mut self) -> std::borrow::Cow<'a, str> {
        String::from_utf8_lossy(self.as_bytes())
    }

    /// convert duckdb_string_t to a byte slice
    pub fn as_bytes(&mut self) -> &'a [u8] {
        unsafe {
            let len = duckdb_string_t_length(*self.ptr);
            let c_ptr = duckdb_string_t_data(self.ptr);
            std::slice::from_raw_parts(c_ptr as *const u8, len as usize)
        }
    }
}

// FIXME: flat vectors don't have all of thsese types. I think they only
/// Converts flat vector to an arrow array
pub fn flat_vector_to_arrow_array(
    vector: &mut Vector,
    len: usize,
) -> Result<Arc<dyn Array>, Box<dyn std::error::Error>> {
    let type_id = vector.logical_type().as_type_id();
    match type_id {
        DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
            let data = vector.as_slice_with_len::<i32>(len);

            Ok(Arc::new(
                PrimitiveArray::<Int32Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S
        | DUCKDB_TYPE::DUCKDB_TYPE_TIME_TZ => {
            let data = vector.as_slice_with_len::<duckdb_timestamp>(len);
            let micros = data.iter().map(|duckdb_timestamp { micros }| *micros);
            let structs = TimestampMicrosecondArray::from_iter_values_with_nulls(
                micros,
                Some(NullBuffer::new(BooleanBuffer::collect_bool(
                    data.len(),
                    |row| !vector.slow_row_is_null(row as u64),
                ))),
            );

            Ok(Arc::new(structs))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR => {
            let data = vector.as_slice_with_len::<duckdb_string_t>(len);

            let duck_strings = data.iter().enumerate().map(|(i, s)| {
                if vector.slow_row_is_null(i as u64) {
                    None
                } else {
                    let mut ptr = *s;
                    Some(DuckString::new(&mut ptr).as_str().to_string())
                }
            });

            let values = duck_strings.collect::<Vec<_>>();

            Ok(Arc::new(StringArray::from(values)))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => {
            let data = vector.as_slice_with_len::<bool>(len);

            Ok(Arc::new(BooleanArray::new(
                BooleanBuffer::from_iter(data.iter().copied()),
                Some(NullBuffer::new(BooleanBuffer::collect_bool(
                    data.len(),
                    |row| !vector.slow_row_is_null(row as u64),
                ))),
            )))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
            let data = vector.as_slice_with_len::<f32>(len);

            Ok(Arc::new(
                PrimitiveArray::<Float32Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
            let data = vector.as_slice_with_len::<f64>(len);

            Ok(Arc::new(
                PrimitiveArray::<Float64Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
            let data = vector.as_slice_with_len::<duckdb_date>(len);

            Ok(Arc::new(Date32Array::from_iter_values_with_nulls(
                data.iter().map(|duckdb_date { days }| *days),
                Some(NullBuffer::new(BooleanBuffer::collect_bool(
                    data.len(),
                    |row| !vector.slow_row_is_null(row as u64),
                ))),
            )))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIME => {
            let data = vector.as_slice_with_len::<duckdb_time>(len);

            Ok(Arc::new(
                PrimitiveArray::<Time64MicrosecondType>::from_iter_values_with_nulls(
                    data.iter().map(|duckdb_time { micros }| *micros),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
            let data = vector.as_slice_with_len::<i16>(len);

            Ok(Arc::new(
                PrimitiveArray::<Int16Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {
            let data = vector.as_slice_with_len::<u16>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt16Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BLOB => {
            let mut data = vector.as_slice_with_len::<duckdb_string_t>(len).to_vec();

            let duck_strings = data.iter_mut().enumerate().map(|(i, ptr)| {
                if vector.slow_row_is_null(i as u64) {
                    None
                } else {
                    Some(DuckString::new(ptr))
                }
            });

            let mut builder = GenericBinaryBuilder::<i32>::new();
            for s in duck_strings {
                if let Some(mut s) = s {
                    builder.append_value(s.as_bytes());
                } else {
                    builder.append_null();
                }
            }

            Ok(Arc::new(builder.finish()))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
            let data = vector.as_slice_with_len::<i8>(len);

            Ok(Arc::new(
                PrimitiveArray::<Int8Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {
            let data = vector.as_slice_with_len::<i64>(len);
            Ok(Arc::new(
                PrimitiveArray::<Int64Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {
            let data = vector.as_slice_with_len::<u64>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt64Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {
            let data = vector.as_slice_with_len::<u8>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt8Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {
            let data = vector.as_slice_with_len::<u32>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt32Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    Some(NullBuffer::new(BooleanBuffer::collect_bool(
                        data.len(),
                        |row| !vector.slow_row_is_null(row as u64),
                    ))),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => {
            // even nano second precision is stored in micros when using the c api
            let data = vector.as_slice_with_len::<duckdb_timestamp>(len);
            let nanos = data
                .iter()
                .map(|duckdb_timestamp { micros }| *micros * 1000);
            let structs = TimestampNanosecondArray::from_iter_values_with_nulls(
                nanos,
                Some(NullBuffer::new(BooleanBuffer::collect_bool(
                    data.len(),
                    |row| !vector.slow_row_is_null(row as u64),
                ))),
            );

            Ok(Arc::new(structs))
        }
        _ => todo!(),
    }
}

pub fn data_chunk_to_arrow(field_names: &FieldNames, chunk: &DataChunk) -> VortexResult<ArrayRef> {
    let len = chunk.len();

    let columns = (0..chunk.column_count())
        .zip(field_names.iter())
        .map(|(i, name)| {
            let mut vector = chunk.get_vector(i);
            vector.flatten(len);
            flat_vector_to_arrow_array(&mut vector, len.as_usize())
                .map(|array_data| {
                    assert_eq!(array_data.len(), chunk.len().as_usize());
                    (name, ArrayRef::from_arrow(array_data.as_ref(), true))
                })
                .map_err(|e| vortex_err!("duckdb to arrow conversion failure {}", e.to_string()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    StructArray::try_from_iter(columns).map(|a| a.to_array())
}
