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
use vortex::dtype::DecimalType;
use vortex::dtype::FieldNames;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::extension::datetime::TimeUnit;
use vortex::mask::Mask;

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
use crate::duckdb::DataChunkRef;
use crate::duckdb::VectorRef;
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
            std::slice::from_raw_parts(c_ptr.cast::<u8>(), len as usize)
        }
    }
}

fn vector_as_slice<T: NativePType>(vector: &VectorRef, len: usize) -> ArrayRef {
    let data = vector.as_slice_with_len::<T>(len);

    PrimitiveArray::new(
        Buffer::copy_from(data),
        vector.validity_ref(data.len()).to_validity(),
    )
    .into_array()
}

fn vector_mapped<T, P: NativePType, F: Fn(&T) -> P>(
    vector: &VectorRef,
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

fn vector_as_string_blob(vector: &VectorRef, len: usize, dtype: DType) -> ArrayRef {
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

/// Converts a valid [`duckdb_list_entry`] to `(offset, size)`, updating tracking state.
///
/// Updates `child_min_length` with the maximum end offset seen so far, and `previous_end` with this
/// entry's end offset (for use as the offset of subsequent null entries).
///
/// Panics if the offset or size are negative or don't fit in the expected types.
fn convert_valid_list_entry(
    entry: &duckdb_list_entry,
    child_min_length: &mut usize,
    previous_end: &mut i64,
) -> (i64, i64) {
    let offset = i64::try_from(entry.offset).vortex_expect("list offset must fit i64");
    assert!(offset >= 0, "list offset must be non-negative");
    let size = i64::try_from(entry.length).vortex_expect("list size must fit i64");
    assert!(size >= 0, "list size must be non-negative");

    let end = usize::try_from(offset + size)
        .vortex_expect("child vector length did not fit into a 32-bit `usize` type");

    *child_min_length = (*child_min_length).max(end);
    *previous_end = offset + size;

    (offset, size)
}

/// Processes DuckDB list entries with validity to produce Vortex-compatible `ListView` offsets and
/// sizes.
///
/// Returns `(offsets, sizes, child_min_length)` where `child_min_length` is the maximum end offset
/// across all valid list entries, used to determine the child vector length coming from DuckDB.
///
/// Null list views in DuckDB may contain garbage offset/size values (which is different from the
/// Arrow specification), so we must check validity before reading them.
///
/// For null entries, we set the offset to the previous list's end and size to 0 so that:
/// 1. We don't accidentally read garbage data from the child vector.
/// 2. The null entries remain in (mostly) sorted offset order, which can potentially simplify
///    downstream operations like converting `ListView` to `List`.
fn process_duckdb_lists(
    entries: &[duckdb_list_entry],
    validity: &Mask,
) -> VortexResult<(Buffer<i64>, Buffer<i64>, usize)> {
    let len = entries.len();
    let mut offsets = BufferMut::with_capacity(len);
    let mut sizes = BufferMut::with_capacity(len);

    match validity {
        Mask::AllTrue(_) => {
            // All entries are valid, so there is no need to check the validity.
            let mut child_min_length = 0;
            let mut previous_end = 0;
            for entry in entries {
                let (offset, size) =
                    convert_valid_list_entry(entry, &mut child_min_length, &mut previous_end);

                // SAFETY: We allocated enough capacity above.
                unsafe {
                    offsets.push_unchecked(offset);
                    sizes.push_unchecked(size);
                }
            }
            Ok((offsets.freeze(), sizes.freeze(), child_min_length))
        }
        Mask::AllFalse(_) => {
            // All entries are null, so we can just set offset=0 and size=0.
            // SAFETY: We allocated enough capacity above.
            unsafe {
                offsets.push_n_unchecked(0, len);
                sizes.push_n_unchecked(0, len);
            }
            Ok((offsets.freeze(), sizes.freeze(), 0))
        }
        Mask::Values(values) => {
            // We have some number of nulls, so make sure to check validity before updating info.
            let mut child_min_length = 0;
            let mut previous_end = 0;

            for (entry, is_valid) in entries.iter().zip(values.bit_buffer().iter()) {
                let (offset, size) = if is_valid {
                    convert_valid_list_entry(entry, &mut child_min_length, &mut previous_end)
                } else {
                    (previous_end, 0)
                };

                // SAFETY: We allocated enough capacity above.
                unsafe {
                    offsets.push_unchecked(offset);
                    sizes.push_unchecked(size);
                }
            }

            Ok((offsets.freeze(), sizes.freeze(), child_min_length))
        }
    }
}

/// Converts flat vector to a vortex array
pub fn flat_vector_to_vortex(vector: &VectorRef, len: usize) -> VortexResult<ArrayRef> {
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
                TemporalArray::new_timestamp(arr, TimeUnit::Microseconds, Some("UTC".into()))
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

            Ok(BoolArray::new(
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
                vector.array_vector_get_child(),
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
            let validity = vector.validity_ref(len).to_mask();
            let entries = vector.as_slice_with_len::<duckdb_list_entry>(len);

            let (offsets, sizes, child_min_length) = process_duckdb_lists(entries, &validity)?;
            let child_data =
                flat_vector_to_vortex(vector.list_vector_get_child(), child_min_length)?;

            ListViewArray::try_new(
                child_data,
                offsets.into_array(),
                sizes.into_array(),
                Validity::from_mask(validity, Nullability::Nullable),
            )
            .map(|a| a.into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_STRUCT => {
            let logical_type = vector.logical_type();
            let children = (0..logical_type.struct_type_child_count())
                .map(|idx| flat_vector_to_vortex(vector.struct_vector_get_child(idx), len))
                .collect::<Result<Vec<_>, _>>()?;
            let names = (0..logical_type.struct_type_child_count())
                .map(|idx| logical_type.struct_child_name(idx))
                .collect();

            StructArray::try_new(names, children, len, vector.validity_ref(len).to_validity())
                .map(|a| a.into_array())
        }
        _ => unimplemented!("missing impl for {type_id:?}"),
    }
}

pub fn data_chunk_to_vortex(
    field_names: &FieldNames,
    chunk: &DataChunkRef,
) -> VortexResult<ArrayRef> {
    let len = chunk.len();

    let columns = (0..chunk.column_count())
        .map(|i| {
            let vector = chunk.get_vector(i);
            vector.flatten(len);
            flat_vector_to_vortex(vector, len.as_())
        })
        .collect::<VortexResult<Arc<_>>>()?;
    StructArray::try_new(
        field_names.clone(),
        columns,
        len.as_(),
        Validity::NonNullable,
    )
    .map(|a| a.into_array())
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use vortex::array::LEGACY_SESSION;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
    use vortex::array::arrays::listview::ListViewArrayExt;
    use vortex::array::arrays::struct_::StructArrayExt;
    use vortex::array::assert_arrays_eq;
    use vortex::error::VortexExpect;
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
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i32>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let expected =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4), Some(5)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_timestamp_vector_conversion() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1_703_980_800_000_000_i64, 0i64, -86_400_000_000_i64]; // microseconds
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
    }

    #[test]
    fn test_timestamp_seconds_vector_conversion() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1_703_980_800_i64, 0i64, -86_400_i64]; // seconds
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S);
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
    }

    #[test]
    fn test_timestamp_milliseconds_vector_conversion() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1_703_980_800_000_i64, 0i64, -86_400_000_i64]; // milliseconds
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS);
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
    }

    #[test]
    fn test_timestamp_with_nulls_conversion() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1_703_980_800_000_000_i64, 0i64, -86_400_000_000_i64];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(&logical_type, len);

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
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
        assert_eq!(
            vortex_values
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(vortex_values.as_ref().len(), &mut ctx)
                .unwrap(),
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
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let vortex_values = vortex_array
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
    }

    #[test]
    fn test_timestamp_single_value() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1_703_980_800_000_000_i64]; // Single microsecond timestamp
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP);
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<i64>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = TemporalArray::try_from(result).unwrap();
        let vortex_values = vortex_array
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let values_slice = vortex_values.as_slice::<i64>();

        assert_eq!(values_slice, values);
    }

    #[test]
    fn test_boolean_vector_conversion() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![true, false, true, false];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN);
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let slice = vector.as_slice_mut::<bool>(len);
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<BoolArray>(&mut ctx).unwrap();
        let expected = BoolArray::new(BitBuffer::from(values), Validity::AllValid);
        assert_arrays_eq!(vortex_array, expected);
    }

    #[test]
    fn test_vector_with_nulls() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1i32, 2, 3];
        let len = values.len();

        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let mut vector = Vector::with_capacity(&logical_type, len);

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
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<PrimitiveArray>(&mut ctx).unwrap();
        let vortex_slice = vortex_array.as_slice::<i32>();

        assert_eq!(vortex_slice, values);
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(vortex_array.as_ref().len(), &mut ctx)
                .unwrap(),
            Mask::from_indices(3, vec![0, 2])
        );
    }

    #[test]
    fn test_list() {
        let values = vec![1i32, 2, 3, 4];
        let len = 1;

        let logical_type =
            LogicalType::list_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER))
                .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let entries = vector.as_slice_mut::<duckdb_list_entry>(len);
            entries[0] = duckdb_list_entry {
                offset: 0,
                length: values.len() as u64,
            };
            let child = vector.list_vector_get_child_mut();
            let slice = child.as_slice_mut::<i32>(values.len());
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let vortex_array = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_arrays_eq!(
            vortex_array.list_elements_at(0).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)])
        );
    }

    #[test]
    fn test_fixed_sized_list() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = vec![1i32, 2, 3, 4];
        let len = 1;

        let logical_type =
            LogicalType::array_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER), 4)
                .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        unsafe {
            let child = vector.array_vector_get_child_mut();
            let slice = child.as_slice_mut::<i32>(values.len());
            slice.copy_from_slice(&values);
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<FixedSizeListArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_arrays_eq!(
            vortex_array.fixed_size_list_elements_at(0).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)])
        );
    }

    #[test]
    fn test_empty_struct() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let len = 4;
        let logical_type = LogicalType::struct_type([], [])
            .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let vector = Vector::with_capacity(&logical_type, len);

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<StructArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(vortex_array.struct_fields().nfields(), 0);
    }

    #[test]
    fn test_struct() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with data
        for (i, values) in
            (0..vector.logical_type().struct_type_child_count()).zip([values1, values2])
        {
            unsafe {
                let child = vector.struct_vector_get_child_mut(i);
                let slice = child.as_slice_mut::<i32>(len);
                slice.copy_from_slice(&values);
            }
        }

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<StructArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(vortex_array.struct_fields().nfields(), 2);
        assert_arrays_eq!(
            vortex_array.unmasked_field(0),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)])
        );
        assert_arrays_eq!(
            vortex_array.unmasked_field(1),
            PrimitiveArray::from_option_iter([Some(5i32), Some(6), Some(7), Some(8)])
        );
    }

    #[test]
    fn test_list_with_trailing_null() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Regression test: when the last list entry is null, its offset/length may be 0/0,
        // so we can't use the last entry to compute child vector length.
        let child_values = vec![1i32, 2, 3, 4];
        let len = 2;

        let logical_type =
            LogicalType::list_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER)).unwrap();
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Entry 0: offset=0, length=4 -> all elements (end=4)
        // Entry 1: null, offset=0, length=0 (end=0)
        unsafe {
            let entries = vector.as_slice_mut::<duckdb_list_entry>(len);
            entries[0] = duckdb_list_entry {
                offset: 0,
                length: child_values.len() as u64,
            };
            entries[1] = duckdb_list_entry {
                offset: 0,
                length: 0,
            };
            let child = vector.list_vector_get_child_mut();
            let slice = child.as_slice_mut::<i32>(child_values.len());
            slice.copy_from_slice(&child_values);
        }

        // Set the second entry as null.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(1, false);

        // Test conversion - the old bug would compute child length as 0+0=0 instead of
        // max(4,0)=4.
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_arrays_eq!(
            vortex_array.list_elements_at(0).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)])
        );
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(vortex_array.as_ref().len(), &mut ctx)
                .unwrap(),
            Mask::from_indices(2, vec![0])
        );
    }

    #[test]
    fn test_list_out_of_order() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Regression test: list views can be out of order in DuckDB. The child vector length
        // must be computed as the maximum end offset, not just the last entry's end offset.
        let child_values = vec![1i32, 2, 3, 4];
        let len = 2;

        let logical_type =
            LogicalType::list_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER)).unwrap();
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Populate with out-of-order list entries:
        // - Entry 0: offset=2, length=2 -> elements [3, 4] (end=4)
        // - Entry 1: offset=0, length=2 -> elements [1, 2] (end=2)
        unsafe {
            let entries = vector.as_slice_mut::<duckdb_list_entry>(len);
            entries[0] = duckdb_list_entry {
                offset: 2,
                length: 2,
            };
            entries[1] = duckdb_list_entry {
                offset: 0,
                length: 2,
            };
            let child = vector.list_vector_get_child_mut();
            let slice = child.as_slice_mut::<i32>(child_values.len());
            slice.copy_from_slice(&child_values);
        }

        // Test conversion - the old bug would compute child length as 0+2=2 instead of
        // max(4,2)=4.
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_arrays_eq!(
            vortex_array.list_elements_at(0).unwrap(),
            PrimitiveArray::from_option_iter([Some(3i32), Some(4)])
        );
        assert_arrays_eq!(
            vortex_array.list_elements_at(1).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2)])
        );
    }

    #[test]
    fn test_list_null_garbage_data() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Test that null list entries with garbage offset/size values don't cause issues.
        // DuckDB doesn't guarantee valid offset/size for null list views, so we must check
        // validity before reading the offset/size values.
        let child_values = vec![1i32, 2, 3, 4];
        let len = 3;

        let logical_type =
            LogicalType::list_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER)).unwrap();
        let mut vector = Vector::with_capacity(&logical_type, len);

        // Entry 0: valid, offset=0, length=2 -> elements [1, 2]
        // Entry 1: null with garbage values (offset=9999, length=9999)
        // Entry 2: valid, offset=2, length=2 -> elements [3, 4]
        unsafe {
            let entries = vector.as_slice_mut::<duckdb_list_entry>(len);
            entries[0] = duckdb_list_entry {
                offset: 0,
                length: 2,
            };
            // Garbage values that would cause a panic if we tried to use them.
            entries[1] = duckdb_list_entry {
                offset: 9999,
                length: 9999,
            };
            entries[2] = duckdb_list_entry {
                offset: 2,
                length: 2,
            };
            let child = vector.list_vector_get_child_mut();
            let slice = child.as_slice_mut::<i32>(child_values.len());
            slice.copy_from_slice(&child_values);
        }

        // Set entry 1 as null.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(1, false);

        // Test conversion. The old code would compute child_min_length as 9999+9999=19998, which
        // would panic when trying to read that much data from the child vector.
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);

        // Valid entries should work correctly.
        assert_arrays_eq!(
            vortex_array.list_elements_at(0).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2)])
        );
        assert_arrays_eq!(
            vortex_array.list_elements_at(2).unwrap(),
            PrimitiveArray::from_option_iter([Some(3i32), Some(4)])
        );

        // Verify the null entry has sanitized offset/size (offset=2, size=0) rather than garbage.
        let offsets = vortex_array
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let sizes = vortex_array
            .sizes()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(offsets.as_slice::<i64>()[1], 2); // Previous end (0+2).
        assert_eq!(sizes.as_slice::<i64>()[1], 0);

        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(vortex_array.as_ref().len(), &mut ctx)
                .unwrap(),
            Mask::from_indices(3, vec![0, 2])
        );
    }
}
