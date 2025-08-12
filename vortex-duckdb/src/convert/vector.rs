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
    Array, BooleanArray, Date32Array, Decimal128Array, PrimitiveArray, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};
use arrow_buffer::buffer::BooleanBuffer;
use bitvec::macros::internal::funty::Fundamental;
use vortex::ArrayRef;
use vortex::arrays::StructArray;
use vortex::arrow::FromArrowArray;
use vortex::dtype::{DecimalDType, FieldNames};
use vortex::error::{VortexResult, vortex_err};
use vortex::scalar::DecimalValueType;

use crate::cpp::{
    DUCKDB_TYPE, duckdb_date, duckdb_string_t, duckdb_string_t_data, duckdb_string_t_length,
    duckdb_time, duckdb_timestamp, duckdb_timestamp_ms, duckdb_timestamp_s,
};
use crate::duckdb::{DataChunk, Vector};
use crate::exporter::precision_to_duckdb_storage_size;

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
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP => {
            let data = vector.as_slice_with_len::<duckdb_timestamp>(len);
            let micros = data.iter().map(|duckdb_timestamp { micros }| *micros);
            let structs = TimestampMicrosecondArray::from_iter_values_with_nulls(
                micros,
                vector.validity_ref(data.len()).to_null_buffer(),
            );

            Ok(Arc::new(structs))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S => {
            let data = vector.as_slice_with_len::<duckdb_timestamp_s>(len);
            let seconds = data.iter().map(|duckdb_timestamp_s { seconds }| *seconds);
            let structs = TimestampSecondArray::from_iter_values_with_nulls(
                seconds,
                vector.validity_ref(data.len()).to_null_buffer(),
            );

            Ok(Arc::new(structs))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS => {
            let data = vector.as_slice_with_len::<duckdb_timestamp_ms>(len);
            let millis = data.iter().map(|duckdb_timestamp_ms { millis }| *millis);
            let structs = TimestampMillisecondArray::from_iter_values_with_nulls(
                millis,
                vector.validity_ref(data.len()).to_null_buffer(),
            );

            Ok(Arc::new(structs))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => {
            let data = vector.as_slice_with_len::<duckdb_timestamp>(len);
            let nanos = data
                .iter()
                .map(|duckdb_timestamp { micros }| *micros * 1000);
            let structs = TimestampNanosecondArray::from_iter_values_with_nulls(
                nanos,
                vector.validity_ref(data.len()).to_null_buffer(),
            );

            Ok(Arc::new(structs))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIME_TZ => {
            let data = vector.as_slice_with_len::<duckdb_timestamp>(len);
            let micros = data.iter().map(|duckdb_timestamp { micros }| *micros);
            let structs = TimestampMicrosecondArray::from_iter_values_with_nulls(
                micros,
                vector.validity_ref(data.len()).to_null_buffer(),
            );

            Ok(Arc::new(structs))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR => {
            let data = vector.as_slice_with_len::<duckdb_string_t>(len);
            let validity = vector.validity_ref(len);

            let duck_strings = data.iter().enumerate().map(|(i, s)| {
                validity.is_valid(i).then(|| {
                    let mut ptr = *s;
                    DuckString::new(&mut ptr).as_str().to_string()
                })
            });

            let values = duck_strings.collect::<Vec<_>>();

            Ok(Arc::new(StringArray::from(values)))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => {
            let data = vector.as_slice_with_len::<bool>(len);

            Ok(Arc::new(BooleanArray::new(
                BooleanBuffer::from_iter(data.iter().copied()),
                vector.validity_ref(data.len()).to_null_buffer(),
            )))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
            let data = vector.as_slice_with_len::<f32>(len);

            Ok(Arc::new(
                PrimitiveArray::<Float32Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
            let data = vector.as_slice_with_len::<f64>(len);

            Ok(Arc::new(
                PrimitiveArray::<Float64Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
            let data = vector.as_slice_with_len::<duckdb_date>(len);

            Ok(Arc::new(Date32Array::from_iter_values_with_nulls(
                data.iter().map(|duckdb_date { days }| *days),
                vector.validity_ref(data.len()).to_null_buffer(),
            )))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIME => {
            let data = vector.as_slice_with_len::<duckdb_time>(len);

            Ok(Arc::new(
                PrimitiveArray::<Time64MicrosecondType>::from_iter_values_with_nulls(
                    data.iter().map(|duckdb_time { micros }| *micros),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
            let data = vector.as_slice_with_len::<i16>(len);

            Ok(Arc::new(
                PrimitiveArray::<Int16Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {
            let data = vector.as_slice_with_len::<u16>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt16Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BLOB => {
            let mut data = vector.as_slice_with_len::<duckdb_string_t>(len).to_vec();
            let validity = vector.validity_ref(len);

            let duck_strings = data
                .iter_mut()
                .enumerate()
                .map(|(i, ptr)| validity.is_valid(i).then(|| DuckString::new(ptr)));

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
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {
            let data = vector.as_slice_with_len::<i64>(len);
            Ok(Arc::new(
                PrimitiveArray::<Int64Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {
            let data = vector.as_slice_with_len::<u64>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt64Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {
            let data = vector.as_slice_with_len::<u8>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt8Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {
            let data = vector.as_slice_with_len::<u32>(len);

            Ok(Arc::new(
                PrimitiveArray::<UInt32Type>::from_iter_values_with_nulls(
                    data.iter().copied(),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL => {
            let logical_type = vector.logical_type();
            let (precision, scale) = logical_type.as_decimal();
            let decimal_dtype = DecimalDType::try_new(precision, scale.try_into()?)?;

            // https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals
            let decimal_values: Vec<i128> = match precision_to_duckdb_storage_size(&decimal_dtype)?
            {
                DecimalValueType::I16 => {
                    let data = vector.as_slice_with_len::<i16>(len);
                    data.iter().map(|&v| v as i128).collect()
                }
                DecimalValueType::I32 => {
                    let data = vector.as_slice_with_len::<i32>(len);
                    data.iter().map(|&v| v as i128).collect()
                }
                DecimalValueType::I64 => {
                    let data = vector.as_slice_with_len::<i64>(len);
                    data.iter().map(|&v| v as i128).collect()
                }
                DecimalValueType::I128 => {
                    let data = vector.as_slice_with_len::<i128>(len);
                    data.to_vec()
                }
                _ => return Err(format!("Unsupported decimal precision: {precision}").into()),
            };

            let decimal_array = Decimal128Array::from_iter_values_with_nulls(
                decimal_values.into_iter(),
                vector.validity_ref(len).to_null_buffer(),
            )
            .with_precision_and_scale(precision, scale as i8)?;

            Ok(Arc::new(decimal_array))
        }
        _ => todo!("missing impl for {:?}", type_id),
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

#[cfg(test)]
mod tests {
    use arrow_array::{
        BooleanArray, Int32Array, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampSecondArray,
    };

    use super::*;
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::{LogicalType, Vector};

    #[test]
    fn test_integer_vector_conversion() {
        let values = vec![1i32, 2, 3, 4, 5];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i32>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(arrow_array.len(), len);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(arrow_array.value(i), expected);
        }
    }

    #[test]
    fn test_timestamp_vector_conversion() {
        let values = vec![1_703_980_800_000_000_i64, 0i64, -86_400_000_000_i64]; // microseconds
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();

        assert_eq!(arrow_array.len(), len);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(arrow_array.value(i), expected);
        }
    }

    #[test]
    fn test_timestamp_seconds_vector_conversion() {
        let values = vec![1_703_980_800_i64, 0i64, -86_400_i64]; // seconds
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .unwrap();

        assert_eq!(arrow_array.len(), len);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(arrow_array.value(i), expected);
        }
    }

    #[test]
    fn test_timestamp_milliseconds_vector_conversion() {
        let values = vec![1_703_980_800_000_i64, 0i64, -86_400_000_i64]; // milliseconds
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();

        assert_eq!(arrow_array.len(), len);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(arrow_array.value(i), expected);
        }
    }

    #[test]
    fn test_timestamp_with_nulls_conversion() {
        let values = vec![1_703_980_800_000_000_i64, 0i64, -86_400_000_000_i64];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Set middle element as null
        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_slice(len) };
        validity_slice.set(1, false);

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();

        assert_eq!(arrow_array.len(), len);
        assert!(arrow_array.is_valid(0));
        assert!(arrow_array.is_null(1));
        assert!(arrow_array.is_valid(2));
        assert_eq!(arrow_array.value(0), values[0]);
        assert_eq!(arrow_array.value(2), values[2]);
    }

    #[test]
    fn test_timestamp_extreme_values() {
        // Test extreme timestamp values
        let values = vec![
            i64::MAX,                       // Maximum possible timestamp
            i64::MIN,                       // Minimum possible timestamp
            0i64,                           // Epoch
            9_223_372_036_854_775_000_i64,  // Near max but reasonable
            -9_223_372_036_854_775_000_i64, // Near min but reasonable
        ];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();

        assert_eq!(arrow_array.len(), len);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(arrow_array.value(i), expected);
        }
    }

    #[test]
    fn test_timestamp_single_value() {
        let values = vec![1_703_980_800_000_000_i64]; // Single microsecond timestamp
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();

        assert_eq!(arrow_array.len(), 1);
        assert_eq!(arrow_array.value(0), values[0]);
    }

    #[test]
    fn test_boolean_vector_conversion() {
        let values = vec![true, false, true, false];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<bool>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(arrow_array.len(), len);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(arrow_array.value(i), expected);
        }
    }

    #[test]
    fn test_vector_with_nulls() {
        let values = vec![1i32, 2, 3];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i32>(len);
            slice.copy_from_slice(&values);
        }

        // Set middle element as null
        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_slice(len) };
        validity_slice.set(1, false);

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(arrow_array.len(), len);
        assert!(arrow_array.is_valid(0));
        assert!(arrow_array.is_null(1));
        assert!(arrow_array.is_valid(2));
        assert_eq!(arrow_array.value(0), 1);
        assert_eq!(arrow_array.value(2), 3);
    }
}
