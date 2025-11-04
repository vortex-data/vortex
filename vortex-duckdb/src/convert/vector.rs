// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Copied of duckdb-rs (https://github.com/duckdb/duckdb-rs/blob/main/crates/duckdb/src/vtab/arrow.rs)
use std::sync::Arc;

use arrow_array::builder::GenericBinaryBuilder;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal128Array, FixedSizeListArray, GenericListViewArray,
    PrimitiveArray, StringArray, Time64MicrosecondArray, Time64NanosecondArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};
use arrow_buffer::buffer::BooleanBuffer;
use arrow_schema::Field;
use num_traits::AsPrimitive;
use vortex::ArrayRef;
use vortex::arrays::StructArray;
use vortex::arrow::FromArrowArray;
use vortex::buffer::BufferMut;
use vortex::dtype::{DType, DecimalDType, FieldNames, Nullability};
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::scalar::DecimalType;
use vortex::validity::Validity;

use crate::convert::dtype::FromLogicalType;
use crate::cpp::{
    DUCKDB_TYPE, duckdb_date, duckdb_list_entry, duckdb_string_t, duckdb_string_t_data,
    duckdb_string_t_length, duckdb_time, duckdb_time_ns, duckdb_timestamp, duckdb_timestamp_ms,
    duckdb_timestamp_s,
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
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ => {
            let data = vector.as_slice_with_len::<duckdb_timestamp>(len);
            let structs = TimestampMicrosecondArray::from_iter_values_with_nulls(
                data.iter().map(|duckdb_timestamp { micros }| *micros),
                vector.validity_ref(data.len()).to_null_buffer(),
            )
            .with_timezone("UTC");

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
                Time64MicrosecondArray::from_iter_values_with_nulls(
                    data.iter().map(|duckdb_time { micros }| *micros),
                    vector.validity_ref(data.len()).to_null_buffer(),
                ),
            ))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIME_NS => {
            let data = vector.as_slice_with_len::<duckdb_time_ns>(len);

            Ok(Arc::new(
                Time64NanosecondArray::from_iter_values_with_nulls(
                    data.iter().map(|duckdb_time_ns { nanos }| *nanos),
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
                DecimalType::I16 => {
                    let data = vector.as_slice_with_len::<i16>(len);
                    data.iter().map(|&v| v as i128).collect()
                }
                DecimalType::I32 => {
                    let data = vector.as_slice_with_len::<i32>(len);
                    data.iter().map(|&v| v as i128).collect()
                }
                DecimalType::I64 => {
                    let data = vector.as_slice_with_len::<i64>(len);
                    data.iter().map(|&v| v as i128).collect()
                }
                DecimalType::I128 => {
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
        DUCKDB_TYPE::DUCKDB_TYPE_ARRAY => {
            let array_elem_size = vector.logical_type().array_type_array_size();
            let array_child_type = vector.logical_type().array_child_type();
            let data_arrow = flat_vector_to_arrow_array(
                &mut vector.array_vector_get_child(),
                len * array_elem_size as usize,
            )?;
            Ok(Arc::new(FixedSizeListArray::try_new(
                Arc::new(Field::new(
                    "element",
                    DType::from_logical_type(array_child_type, Nullability::Nullable)?
                        .to_arrow_dtype()?,
                    true,
                )),
                array_elem_size as i32,
                data_arrow,
                vector.validity_ref(len).to_null_buffer(),
            )?))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_LIST => {
            let array_child_type = vector.logical_type().list_child_type();

            let mut offsets = BufferMut::with_capacity(len);
            let mut lengths = BufferMut::with_capacity(len);
            for duckdb_list_entry { offset, length } in
                vector.as_slice_with_len::<duckdb_list_entry>(len)
            {
                unsafe {
                    offsets.push_unchecked(
                        i64::try_from(*offset).vortex_expect("offset must fit i64"),
                    );
                    lengths.push_unchecked(
                        i64::try_from(*length).vortex_expect("length must fit i64"),
                    );
                }
            }
            let offsets = offsets.freeze();
            let lengths = lengths.freeze();
            let arrow_child = flat_vector_to_arrow_array(
                &mut vector.list_vector_get_child(),
                usize::try_from(offsets[len - 1] + lengths[len - 1])
                    .vortex_expect("last offset and length sum must fit in usize "),
            )?;

            Ok(Arc::new(GenericListViewArray::try_new(
                Arc::new(Field::new(
                    "element",
                    DType::from_logical_type(array_child_type, Nullability::Nullable)?
                        .to_arrow_dtype()?,
                    true,
                )),
                offsets.into_arrow_scalar_buffer(),
                lengths.into_arrow_scalar_buffer(),
                arrow_child,
                vector.validity_ref(len).to_null_buffer(),
            )?))
        }
        DUCKDB_TYPE::DUCKDB_TYPE_STRUCT => {
            let children = (0..vector.logical_type().struct_type_child_count())
                .map(|idx| {
                    flat_vector_to_arrow_array(&mut vector.struct_vector_get_child(idx), len)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if children.is_empty() {
                Ok(Arc::new(arrow_array::StructArray::new_empty_fields(
                    len,
                    vector.validity_ref(len).to_null_buffer(),
                )))
            } else {
                Ok(Arc::new(arrow_array::StructArray::try_new(
                    DType::from_logical_type(vector.logical_type(), Nullability::NonNullable)?
                        .to_arrow_schema()?
                        .fields,
                    children,
                    vector.validity_ref(len).to_null_buffer(),
                )?))
            }
        }
        _ => todo!("missing impl for {type_id:?}"),
    }
}

pub fn data_chunk_to_arrow(field_names: &FieldNames, chunk: &DataChunk) -> VortexResult<ArrayRef> {
    let len = chunk.len();

    let columns = (0..chunk.column_count())
        .map(|i| {
            let mut vector = chunk.get_vector(i);
            vector.flatten(len);
            flat_vector_to_arrow_array(&mut vector, len.as_())
                .map(|array_data| {
                    let chunk_len: usize = chunk.len().as_();
                    assert_eq!(array_data.len(), chunk_len);
                    ArrayRef::from_arrow(array_data.as_ref(), true)
                })
                .map_err(|e| vortex_err!("duckdb to arrow conversion failure {e}"))
        })
        .collect::<VortexResult<Arc<_>>>()?;
    StructArray::try_new(
        field_names.clone(),
        columns,
        len.as_(),
        Validity::NonNullable,
    )
    .map(|a| a.to_array())
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        BooleanArray, Int32Array, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampSecondArray,
    };
    use vortex::error::VortexUnwrap;

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
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
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
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
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

    #[test]
    fn test_list() {
        let values = vec![1i32, 2, 3, 4];
        let len = 1;

        let logical_type =
            LogicalType::list_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER))
                .vortex_unwrap();
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let entries = vector.as_slice_mut::<duckdb_list_entry>(len);
            entries[0] = duckdb_list_entry {
                offset: 0,
                length: values.len() as u64,
            };
            let mut child = vector.list_vector_get_child();
            let slice = child.as_slice_mut::<i32>(values.len());
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_list_view::<i64>();

        assert_eq!(arrow_array.len(), len);
        assert_eq!(
            arrow_array.value(0).as_primitive::<Int32Type>(),
            &Int32Array::from_iter([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_fixed_sized_list() {
        let values = vec![1i32, 2, 3, 4];
        let len = 1;

        let logical_type =
            LogicalType::array_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER), 4)
                .vortex_unwrap();
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        unsafe {
            let mut child = vector.array_vector_get_child();
            let slice = child.as_slice_mut::<i32>(values.len());
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_fixed_size_list();

        assert_eq!(arrow_array.len(), len);
        assert_eq!(
            arrow_array.value(0).as_primitive::<Int32Type>(),
            &Int32Array::from_iter([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_empty_struct() {
        let len = 4;
        let logical_type = LogicalType::struct_type([], []).vortex_unwrap();
        let mut vector = Vector::with_capacity(logical_type, len);

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_struct();

        assert_eq!(arrow_array.len(), len);
        assert_eq!(arrow_array.fields().len(), 0);
    }

    #[test]
    fn test_struct() {
        let values1 = vec![1i32, 2, 3, 4];
        let values2 = vec![5i32, 6, 7, 8];
        let len = values1.len();

        let logical_type = LogicalType::struct_type(
            [
                LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER),
                LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER),
            ],
            [CString::new("a").unwrap(), CString::new("b").unwrap()],
        )
        .vortex_unwrap();
        let mut vector = Vector::with_capacity(logical_type, len);

        // Populate with data
        for (i, values) in
            (0..vector.logical_type().struct_type_child_count()).zip([values1, values2])
        {
            unsafe {
                let mut child = vector.struct_vector_get_child(i);
                let slice = child.as_slice_mut::<i32>(len);
                slice.copy_from_slice(&values);
            }
        }

        // Test conversion
        let result = flat_vector_to_arrow_array(&mut vector, len).unwrap();
        let arrow_array = result.as_struct();

        assert_eq!(arrow_array.len(), len);
        assert_eq!(arrow_array.fields().len(), 2);
        assert_eq!(
            arrow_array.column(0).as_primitive::<Int32Type>(),
            &Int32Array::from_iter([1, 2, 3, 4])
        );
        assert_eq!(
            arrow_array.column(1).as_primitive::<Int32Type>(),
            &Int32Array::from_iter([5, 6, 7, 8])
        );
    }
}
