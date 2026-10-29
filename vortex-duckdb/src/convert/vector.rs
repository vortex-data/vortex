// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::decimal::check_precision;
use vortex::array::arrays::decimal::copy_to_canonical_width;
use vortex::array::buffer::BufferHandle;
use vortex::array::builders::ArrayBuilder;
use vortex::array::builders::VarBinViewBuilder;
use vortex::array::dtype::extension::ExtDType;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::BufferAllocatorRef;
use vortex::buffer::BufferMut;
use vortex::buffer::trusted_len::TrustedLen;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::FieldNames;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::encodings::uuid::Uuid;
use vortex::encodings::uuid::UuidMetadata;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::extension::datetime::TimeUnit;
use vortex::mask::Mask;
use vortex_spatial::extension::SpatialMetadata;
use vortex_spatial::extension::WellKnownBinary;

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

fn vector_i128_values(vector: &VectorRef, len: usize) -> impl TrustedLen<Item = i128> {
    let base = unsafe { crate::cpp::duckdb_vector_get_data(vector.as_ptr()) }.cast::<i128>();
    // In batch copy, columns are 8 byte aligned, so reading vector's values
    // as &[i128] is UB. Read data unaligned specifically
    (0..len).map(move |i| unsafe { base.add(i).read_unaligned() })
}

/// Copies the values of a DuckDB decimal vector into the canonical width of `decimal_dtype`, and
/// checks that every valid value is within its precision.
///
/// DuckDB stores precision 1 to 4 as `i16`, so precision 1 to 2 narrows into `i8`, and
/// [`copy_to_canonical_width`] checks that copy. Every other DuckDB width is the canonical width,
/// where the copy checks nothing, so [`check_precision`] checks the values first.
fn copy_decimal_values<T: NativeDecimalType>(
    values: &[T],
    validity: &Mask,
    decimal_dtype: DecimalDType,
) -> VortexResult<BufferHandle> {
    if T::DECIMAL_TYPE == decimal_dtype.canonical_values_type() {
        check_precision(values, validity, decimal_dtype)?;
    }

    copy_to_canonical_width(values, validity, decimal_dtype)
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

    let mut builder =
        VarBinViewBuilder::with_capacity_in(dtype, len, BufferAllocatorRef::statically_allocated());

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
    let logical_type = vector.logical_type();
    match logical_type.as_type_id() {
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
        DUCKDB_TYPE::DUCKDB_TYPE_GEOMETRY => {
            let wkb_values =
                vector_as_string_blob(vector, len, DType::Binary(Nullability::Nullable));
            let crs = logical_type.geometry_crs().map(|crs| crs.to_string());
            let wkb_type = ExtDType::<WellKnownBinary>::try_new(
                SpatialMetadata::new(crs),
                DType::Binary(Nullability::Nullable),
            )?
            .erased();

            Ok(ExtensionArray::try_new(wkb_type, wkb_values.into_array())?.into_array())
        }
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
            let mask = vector.validity_ref(len).execute_mask();

            // https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals
            let values = match precision_to_duckdb_storage_size(&decimal_dtype)? {
                DecimalType::I16 => {
                    let data = vector.as_slice_with_len::<i16>(len);
                    copy_decimal_values(data, &mask, decimal_dtype)?
                }
                DecimalType::I32 => {
                    let data = vector.as_slice_with_len::<i32>(len);
                    copy_decimal_values(data, &mask, decimal_dtype)?
                }
                DecimalType::I64 => {
                    let data = vector.as_slice_with_len::<i64>(len);
                    copy_decimal_values(data, &mask, decimal_dtype)?
                }
                DecimalType::I128 => {
                    // DuckDB columns can be misaligned for `i128`, so `vector_i128_values` reads
                    // them into a new buffer, which already has the canonical width of precision
                    // 19 to 38.
                    let data = Buffer::from_iter(vector_i128_values(vector, len));
                    check_precision(data.as_slice(), &mask, decimal_dtype)?;
                    BufferHandle::new_host(data.into_byte_buffer())
                }
                _ => vortex_bail!("Unsupported decimal precision: {precision}"),
            };
            let validity = Validity::from_mask(mask, Nullability::Nullable);

            // SAFETY: `copy_decimal_values` and the `I128` arm checked that every valid value is
            // within the precision of `decimal_dtype`. The values have the canonical width, and
            // they hold `len` values, as the validity does.
            let array = unsafe {
                DecimalArray::new_unchecked_handle(
                    values,
                    decimal_dtype.canonical_values_type(),
                    decimal_dtype,
                    validity,
                )
            };

            Ok(array.into_array())
        }
        DUCKDB_TYPE::DUCKDB_TYPE_UUID => {
            let mut bytes = BufferMut::<u8>::with_capacity(len * 16);
            for v in vector_i128_values(vector, len) {
                let be = (v as u128) ^ (1u128 << 127);
                bytes.extend_from_slice(&be.to_be_bytes());
            }

            let storage = FixedSizeListArray::try_new(
                PrimitiveArray::new(bytes.freeze(), Validity::NonNullable).into_array(),
                16,
                vector.validity_ref(len).to_validity(),
                len,
            )?;
            let ext_dtype = ExtDType::<Uuid>::try_with_vtable(
                Uuid,
                UuidMetadata::default(),
                storage.dtype().clone(),
            )?
            .erased();

            Ok(ExtensionArray::try_new(ext_dtype, storage.into_array())?.into_array())
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
            let validity = vector.validity_ref(len).execute_mask();
            let entries = vector.as_slice_with_len::<duckdb_list_entry>(len);

            let (offsets, sizes, child_min_length) = process_duckdb_lists(entries, &validity)?;
            let child_data =
                flat_vector_to_vortex(vector.list_vector_get_child(), child_min_length)?;

            // SAFETY: the offsets and sizes are both `i64`, and the child is converted with the
            // largest `offset + size`, so every list lies within it. The validity has one entry
            // per list.
            Ok(unsafe {
                ListViewArray::new_unchecked(
                    child_data,
                    offsets.into_array(),
                    sizes.into_array(),
                    Validity::from_mask(validity, Nullability::Nullable),
                )
            }
            .into_array())
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
        type_id => {
            vortex_bail!("{type_id:?} flat Vector to Vortex array not supported")
        }
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
            vector.flatten();
            flat_vector_to_vortex(vector, len.as_())
        })
        .collect::<VortexResult<Vec<_>>>()?;
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
    use std::sync::LazyLock;

    use geo_types::point;
    use rstest::rstest;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::Decimal;
    use vortex::array::arrays::Extension;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
    use vortex::array::arrays::listview::ListViewArrayExt;
    use vortex::array::arrays::listview::ListViewArraySlotsExt;
    use vortex::array::arrays::struct_::StructArrayExt;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::BigCast;
    use vortex::error::VortexExpect;
    use vortex::mask::Mask;
    use vortex::session::VortexSession;
    use vortex_spatial::extension::WellKnownBinaryData;
    use wkb::writer::WriteOptions;
    use wkb::writer::write_point;

    use super::*;
    use crate::cpp;
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::LogicalType;
    use crate::duckdb::Vector;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Builds a DuckDB decimal vector of `precision` and scale 0 that holds `slots`, and marks
    /// `null_rows` as null. Each slot is written at the DuckDB storage width of `precision`.
    fn decimal_vector(precision: u8, slots: &[i128], null_rows: &[usize]) -> VortexResult<Vector> {
        let len = slots.len();
        let logical_type = LogicalType::decimal_type(precision, 0)?;
        let mut vector = Vector::with_capacity(&logical_type, len);

        match precision_to_duckdb_storage_size(&DecimalDType::try_new(precision, 0)?)? {
            DecimalType::I16 => write_decimal_slots::<i16>(&mut vector, slots),
            DecimalType::I32 => write_decimal_slots::<i32>(&mut vector, slots),
            DecimalType::I64 => write_decimal_slots::<i64>(&mut vector, slots),
            DecimalType::I128 => write_decimal_slots::<i128>(&mut vector, slots),
            storage => unreachable!("DuckDB stores decimals in 16 to 128 bits, got {storage}"),
        }

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        for &row in null_rows {
            validity_slice.set(row, false);
        }

        Ok(vector)
    }

    fn write_decimal_slots<T: NativeDecimalType>(vector: &mut Vector, slots: &[i128]) {
        // SAFETY: Vector was created with a capacity of `slots.len()`.
        let storage = unsafe { vector.as_slice_mut::<T>(slots.len()) };
        for (storage, &slot) in storage.iter_mut().zip(slots) {
            *storage = <T as BigCast>::from(slot)
                .vortex_expect("test slot must fit the DuckDB storage width");
        }
    }

    #[test]
    fn test_integer_vector_conversion() {
        let mut ctx = SESSION.create_execution_ctx();
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
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_timestamp_vector_conversion() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array(),
            TimeUnit::Microseconds,
            None,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_timestamp_seconds_vector_conversion() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array(),
            TimeUnit::Seconds,
            None,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_timestamp_milliseconds_vector_conversion() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_timestamp_with_nulls_conversion() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter([
                Some(1_703_980_800_000_000_i64),
                None,
                Some(-86_400_000_000_i64),
            ])
            .into_array(),
            TimeUnit::Microseconds,
            None,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
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
        let mut ctx = SESSION.create_execution_ctx();
        let expected = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array(),
            TimeUnit::Microseconds,
            None,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_timestamp_single_value() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array(),
            TimeUnit::Microseconds,
            None,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_boolean_vector_conversion() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = BoolArray::new(BitBuffer::from(values), Validity::AllValid);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_vector_with_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let expected = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        assert_arrays_eq!(result, expected, &mut ctx);
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
        let mut ctx = SESSION.create_execution_ctx();
        let vortex_array = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_arrays_eq!(
            vortex_array.list_elements_at(0, &mut ctx).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)]),
            &mut ctx
        );
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(len, &mut ctx)
                .unwrap(),
            Mask::AllTrue(len)
        );
    }

    #[test]
    fn test_fixed_sized_list() {
        let mut ctx = SESSION.create_execution_ctx();
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
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)]),
            &mut ctx
        );
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(len, &mut ctx)
                .unwrap(),
            Mask::AllTrue(len)
        );
    }

    #[test]
    fn test_empty_struct() {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 4;
        let logical_type = LogicalType::struct_type([], [])
            .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let vector = Vector::with_capacity(&logical_type, len);

        // Test conversion
        let result = flat_vector_to_vortex(&vector, len).unwrap();
        let vortex_array = result.execute::<StructArray>(&mut ctx).unwrap();

        assert_eq!(vortex_array.len(), len);
        assert_eq!(vortex_array.struct_fields().nfields(), 0);
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(len, &mut ctx)
                .unwrap(),
            Mask::AllTrue(len)
        );
    }

    #[test]
    fn test_struct() {
        let mut ctx = SESSION.create_execution_ctx();
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
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)]),
            &mut ctx
        );
        assert_arrays_eq!(
            vortex_array.unmasked_field(1),
            PrimitiveArray::from_option_iter([Some(5i32), Some(6), Some(7), Some(8)]),
            &mut ctx
        );
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(len, &mut ctx)
                .unwrap(),
            Mask::AllTrue(len)
        );
        assert_eq!(
            vortex_array
                .struct_fields()
                .names()
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<&str>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn test_list_with_trailing_null() {
        let mut ctx = SESSION.create_execution_ctx();
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
            vortex_array.list_elements_at(0, &mut ctx).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4)]),
            &mut ctx
        );
        assert_eq!(
            vortex_array
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(vortex_array.as_ref().len(), &mut ctx)
                .unwrap(),
            Mask::from_indices(2, vec![0])
        );
    }

    #[test]
    fn test_list_out_of_order() {
        let mut ctx = SESSION.create_execution_ctx();
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
            vortex_array.list_elements_at(0, &mut ctx).unwrap(),
            PrimitiveArray::from_option_iter([Some(3i32), Some(4)]),
            &mut ctx
        );
        assert_arrays_eq!(
            vortex_array.list_elements_at(1, &mut ctx).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_list_null_garbage_data() {
        let mut ctx = SESSION.create_execution_ctx();
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
            vortex_array.list_elements_at(0, &mut ctx).unwrap(),
            PrimitiveArray::from_option_iter([Some(1i32), Some(2)]),
            &mut ctx
        );
        assert_arrays_eq!(
            vortex_array.list_elements_at(2, &mut ctx).unwrap(),
            PrimitiveArray::from_option_iter([Some(3i32), Some(4)]),
            &mut ctx
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
                .execute_mask(vortex_array.as_ref().len(), &mut ctx)
                .unwrap(),
            Mask::from_indices(3, vec![0, 2])
        );
    }

    #[test]
    fn test_geometry_vector_conversion() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let mut wkb_a: Vec<u8> = Vec::new();
        write_point(
            &mut wkb_a,
            &point!(x: 1.0_f64, y: 2.0_f64),
            &WriteOptions::default(),
        )
        .map_err(|e| vortex::error::vortex_err!("writing WKB point: {e}"))?;
        let mut wkb_b: Vec<u8> = Vec::new();
        write_point(
            &mut wkb_b,
            &point!(x: 3.5_f64, y: -4.25_f64),
            &WriteOptions::default(),
        )
        .map_err(|e| vortex::error::vortex_err!("writing WKB point: {e}"))?;

        let len = 3;
        let logical_type = LogicalType::geometry_type(Some("EPSG:4326"))?;
        let mut vector = Vector::with_capacity(&logical_type, len);

        // WKB contains embedded null bytes, so use the length-aware assignment.
        unsafe {
            cpp::duckdb_vector_assign_string_element_len(
                vector.as_ptr(),
                0,
                wkb_a.as_ptr().cast(),
                wkb_a.len() as _,
            );
            cpp::duckdb_vector_assign_string_element_len(
                vector.as_ptr(),
                2,
                wkb_b.as_ptr().cast(),
                wkb_b.len() as _,
            );
        }

        // SAFETY: Vector was created with this length.
        let validity_slice = unsafe { vector.ensure_validity_bitslice(len) };
        validity_slice.set(1, false);

        let result = flat_vector_to_vortex(&vector, len)?;
        let extension = result
            .as_opt::<Extension>()
            .vortex_expect("expected ExtensionArray")
            .into_owned();
        let wkb_data = WellKnownBinaryData::try_from(extension)?;

        assert_eq!(wkb_data.spatial_metadata().get_crs(), Some("EPSG:4326"));

        let storage = wkb_data.wkb_values();
        let expected = VarBinViewArray::from_iter_nullable_bin([
            Some(wkb_a.as_slice()),
            None,
            Some(wkb_b.as_slice()),
        ]);
        assert_arrays_eq!(storage, expected, &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::p1(1, DecimalType::I8)]
    #[case::p2(2, DecimalType::I8)]
    #[case::p3(3, DecimalType::I16)]
    #[case::p4(4, DecimalType::I16)]
    #[case::p9(9, DecimalType::I32)]
    #[case::p18(18, DecimalType::I64)]
    #[case::p38(38, DecimalType::I128)]
    fn test_decimal_vector_conversion(
        #[case] precision: u8,
        #[case] values_type: DecimalType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // The largest and smallest values of the precision, around a null and a zero.
        let max = 10i128.pow(precision.into()) - 1;
        let vector = decimal_vector(precision, &[-max, 0, 0, max], &[1])?;

        let result = flat_vector_to_vortex(&vector, 4)?;
        assert_eq!(result.as_::<Decimal>().values_type(), values_type);

        let expected = DecimalArray::from_option_iter(
            [Some(-max), None, Some(0), Some(max)],
            DecimalDType::new(precision, 0),
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_decimal_vector_null_slot_garbage() -> VortexResult<()> {
        // 300 fits neither precision 2 nor `i8`, but the import must not check a null slot.
        let vector = decimal_vector(2, &[42, 300, -42], &[1])?;

        let result = flat_vector_to_vortex(&vector, 3)?;
        let decimal = result.as_::<Decimal>();
        assert_eq!(decimal.values_type(), DecimalType::I8);
        assert_eq!(decimal.buffer::<i8>().as_slice(), &[42, 0, -42]);
        Ok(())
    }

    #[test]
    fn test_decimal_vector_value_outside_precision() -> VortexResult<()> {
        // Precision 2 narrows the DuckDB `i16` storage into `i8`.
        let vector = decimal_vector(2, &[42, 150], &[])?;

        let err = flat_vector_to_vortex(&vector, 2)
            .expect_err("a valid value outside the precision must fail the import");
        let message = err.to_string();
        assert!(message.contains("row 1"), "{message}");
        Ok(())
    }

    /// Each precision keeps the DuckDB storage width, and `outside` fits that width but not the
    /// precision.
    #[rstest]
    #[case::p4_i16(4, 10_000)]
    #[case::p20_i128(20, 10i128.pow(20))]
    fn test_decimal_vector_equal_width_outside_precision(
        #[case] precision: u8,
        #[case] outside: i128,
    ) -> VortexResult<()> {
        let vector = decimal_vector(precision, &[42, outside], &[])?;

        let err = flat_vector_to_vortex(&vector, 2)
            .expect_err("a valid value outside the precision must fail the import");
        let message = err.to_string();
        assert!(message.contains("row 1"), "{message}");
        Ok(())
    }
}
