// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::builders::ArrayBuilder;
use vortex::array::builders::VarBinViewBuilder;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldNames;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::dtype::datetime::TimeUnit;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::scalar::DecimalType;

use crate::cpp::DUCKDB_TYPE;
use crate::cpp::duckdb_date;
use crate::cpp::duckdb_list_entry;
use crate::cpp::duckdb_string_t;
use crate::cpp::duckdb_string_t_data;
use crate::cpp::duckdb_string_t_length;
use crate::cpp::duckdb_time;
use crate::cpp::duckdb_time_ns;
use crate::cpp::duckdb_timestamp;
use crate::cpp::duckdb_timestamp_ms;
use crate::cpp::duckdb_timestamp_ns;
use crate::cpp::duckdb_timestamp_s;
use crate::duckdb::DataChunk;
use crate::duckdb::Vector;
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
    /// convert duckdb_string_t to a byte slice
    pub fn as_bytes(&mut self) -> &'a [u8] {
        unsafe {
            let len = duckdb_string_t_length(*self.ptr);
            let c_ptr = duckdb_string_t_data(self.ptr);
            std::slice::from_raw_parts(c_ptr as *const u8, len as usize)
        }
    }
}

fn vector_as_slice<T: NativePType>(vector: &mut Vector, len: usize) -> ArrayRef {
    let data = vector.as_slice_with_len::<T>(len);

    PrimitiveArray::new(
        Buffer::copy_from(data),
        vector.validity_ref(data.len()).to_validity(),
    )
    .into_array()
}

fn vector_mapped<T, P: NativePType, F: Fn(&T) -> P>(
    vector: &mut Vector,
    len: usize,
    from_duckdb_type: F,
) -> ArrayRef {
    let data = vector.as_slice_with_len::<T>(len);
    let micros = data.iter().map(from_duckdb_type);
    PrimitiveArray::new(
        Buffer::from_trusted_len_iter(micros),
        vector.validity_ref(data.len()).to_validity(),
    )
    .into_array()
}

fn vector_as_string_blob(vector: &mut Vector, len: usize, dtype: DType) -> ArrayRef {
    let data = vector.as_slice_with_len::<duckdb_string_t>(len);
    let validity = vector.validity_ref(len);

    let mut builder = VarBinViewBuilder::with_capacity(dtype, len);

    for (i, s) in data.iter().enumerate() {
        if validity.is_valid(i) {
            let mut ptr = *s;
            builder.append_value(DuckString::new(&mut ptr).as_bytes())
        } else {
            builder.append_null()
        }
    }

    builder.finish()
}

/// Converts flat vector to a vortex array
pub fn flat_vector_to_vortex(vector: &mut Vector, len: usize) -> VortexResult<ArrayRef> {
    let type_id = vector.logical_type().as_type_id();
    match type_id {
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP => {
            let arr = vector_mapped(vector, len, |duckdb_timestamp { micros }| *micros);
            Ok(TemporalArray::new_timestamp(arr, TimeUnit::Microseconds, None).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S => {
            let arr = vector_mapped(vector, len, |duckdb_timestamp_s { seconds }| *seconds);
            Ok(TemporalArray::new_timestamp(arr, TimeUnit::Seconds, None).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS => {
            let arr = vector_mapped(vector, len, |duckdb_timestamp_ms { millis }| *millis);
            Ok(TemporalArray::new_timestamp(arr, TimeUnit::Milliseconds, None).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => {
            let arr = vector_mapped(vector, len, |duckdb_timestamp_ns { nanos }| *nanos);
            Ok(TemporalArray::new_timestamp(arr, TimeUnit::Nanoseconds, None).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ => {
            let arr = vector_mapped(vector, len, |duckdb_timestamp { micros }| *micros);
            Ok(
                TemporalArray::new_timestamp(arr, TimeUnit::Microseconds, Some("UTC".to_string()))
                    .into_array(),
            )
        }
        DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
            let arr = vector_mapped(vector, len, |duckdb_date { days }| *days);
            Ok(TemporalArray::new_date(arr, TimeUnit::Days).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIME => {
            let arr = vector_mapped(vector, len, |duckdb_time { micros }| *micros);
            Ok(TemporalArray::new_time(arr, TimeUnit::Microseconds).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TIME_NS => {
            let arr = vector_mapped(vector, len, |duckdb_time_ns { nanos }| *nanos);
            Ok(TemporalArray::new_time(arr, TimeUnit::Nanoseconds).into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR => Ok(vector_as_string_blob(
            vector,
            len,
            DType::Utf8(Nullability::Nullable),
        )),
        DUCKDB_TYPE::DUCKDB_TYPE_BLOB => Ok(vector_as_string_blob(
            vector,
            len,
            DType::Binary(Nullability::Nullable),
        )),
        DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => {
            let data = vector.as_slice_with_len::<bool>(len);

            Ok(BoolArray::from_bit_buffer(
                BitBuffer::from(data),
                vector.validity_ref(data.len()).to_validity(),
            )
            .into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => Ok(vector_as_slice::<i8>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => Ok(vector_as_slice::<i16>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => Ok(vector_as_slice::<i32>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => Ok(vector_as_slice::<i64>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => Ok(vector_as_slice::<u8>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => Ok(vector_as_slice::<u16>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => Ok(vector_as_slice::<u32>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => Ok(vector_as_slice::<u64>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => Ok(vector_as_slice::<f32>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => Ok(vector_as_slice::<f64>(vector, len)),
        DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL => {
            let logical_type = vector.logical_type();
            let (precision, scale) = logical_type.as_decimal();
            let decimal_dtype = DecimalDType::try_new(precision, scale.try_into()?)?;
            let validity = vector.validity_ref(len).to_validity();

            // https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals
            match precision_to_duckdb_storage_size(&decimal_dtype)? {
                DecimalType::I16 => {
                    let data = vector.as_slice_with_len::<i16>(len);
                    DecimalArray::try_new(Buffer::copy_from(data), decimal_dtype, validity)
                }
                DecimalType::I32 => {
                    let data = vector.as_slice_with_len::<i32>(len);
                    DecimalArray::try_new(Buffer::copy_from(data), decimal_dtype, validity)
                }
                DecimalType::I64 => {
                    let data = vector.as_slice_with_len::<i64>(len);
                    DecimalArray::try_new(Buffer::copy_from(data), decimal_dtype, validity)
                }
                DecimalType::I128 => {
                    let data = vector.as_slice_with_len::<i128>(len);
                    DecimalArray::try_new(Buffer::copy_from(data), decimal_dtype, validity)
                }
                _ => vortex_bail!("Unsupported decimal precision: {precision}"),
            }
            .map(|a| a.into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_ARRAY => {
            let array_elem_size = vector.logical_type().array_type_array_size();
            let child_data = flat_vector_to_vortex(
                &mut vector.array_vector_get_child(),
                len * array_elem_size as usize,
            )?;

            FixedSizeListArray::try_new(
                child_data,
                array_elem_size,
                vector.validity_ref(len).to_validity(),
                len,
            )
            .map(|a| a.into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_LIST => {
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
            let child_data = flat_vector_to_vortex(
                &mut vector.list_vector_get_child(),
                usize::try_from(offsets[len - 1] + lengths[len - 1])
                    .vortex_expect("last offset and length sum must fit in usize "),
            )?;

            ListViewArray::try_new(
                child_data,
                offsets.into_array(),
                lengths.into_array(),
                vector.validity_ref(len).to_validity(),
            )
            .map(|a| a.into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_STRUCT => {
            let logical_type = vector.logical_type();
            let children = (0..logical_type.struct_type_child_count())
                .map(|idx| flat_vector_to_vortex(&mut vector.struct_vector_get_child(idx), len))
                .collect::<Result<Vec<_>, _>>()?;
            let names = (0..logical_type.struct_type_child_count())
                .map(|idx| logical_type.struct_child_name(idx))
                .collect();

            StructArray::try_new(names, children, len, vector.validity_ref(len).to_validity())
                .map(|a| a.into_array())
        }
        _ => todo!("missing impl for {type_id:?}"),
    }
}

pub fn data_chunk_to_vortex(field_names: &FieldNames, chunk: &DataChunk) -> VortexResult<ArrayRef> {
    let len = chunk.len();

    let columns = (0..chunk.column_count())
        .map(|i| {
            let mut vector = chunk.get_vector(i);
            vector.flatten(len);
            flat_vector_to_vortex(&mut vector, len.as_())
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

    use vortex::array::ToCanonical;
    use vortex::array::arrays::PrimitiveVTable;
    use vortex::error::VortexUnwrap;
    use vortex::mask::Mask;

    use super::*;
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::LogicalType;
    use crate::duckdb::Vector;

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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.as_::<PrimitiveVTable>().as_slice::<i32>();

        assert_eq!(vortex_array, values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array.temporal_values().to_primitive();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array.temporal_values().to_primitive();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array.temporal_values().to_primitive();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array.temporal_values().to_primitive();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
        assert_eq!(
            vortex_values.validity_mask(),
            Mask::from_indices(3, vec![0, 2])
        );
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array.temporal_values().to_primitive();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array.temporal_values().to_primitive();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.to_bool();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(vortex_array.bit_buffer().iter().collect::<Vec<_>>(), values);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.to_primitive();
        let vortex_slice = vortex_array.as_slice::<i32>();

        assert_eq!(vortex_slice, values);
        assert_eq!(
            vortex_array.validity_mask(),
            Mask::from_indices(3, vec![0, 2])
        );
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.to_listview();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(
            vortex_array
                .list_elements_at(0)
                .to_primitive()
                .as_slice::<i32>(),
            &[1, 2, 3, 4]
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.to_fixed_size_list();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(
            vortex_array
                .fixed_size_list_elements_at(0)
                .to_primitive()
                .as_slice::<i32>(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn test_empty_struct() {
        let len = 4;
        let logical_type = LogicalType::struct_type([], []).vortex_unwrap();
        let mut vector = Vector::with_capacity(logical_type, len);

        // Test conversion
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.to_struct();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(vortex_array.fields().len(), 0);
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
        let result = flat_vector_to_vortex(&mut vector, len).unwrap();
        let vortex_array = result.to_struct();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(vortex_array.fields().len(), 2);
        assert_eq!(
            vortex_array.fields()[0].to_primitive().as_slice::<i32>(),
            &[1, 2, 3, 4]
        );
        assert_eq!(
            vortex_array.fields()[1].to_primitive().as_slice::<i32>(),
            &[5, 6, 7, 8]
        );
    }
}
