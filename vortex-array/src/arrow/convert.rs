// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::AnyDictionaryArray;
use arrow_array::Array as ArrowArray;
use arrow_array::ArrowPrimitiveType;
use arrow_array::BooleanArray as ArrowBooleanArray;
use arrow_array::DictionaryArray;
use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
use arrow_array::GenericByteArray;
use arrow_array::GenericByteViewArray;
use arrow_array::GenericListArray;
use arrow_array::GenericListViewArray;
use arrow_array::NullArray as ArrowNullArray;
use arrow_array::OffsetSizeTrait;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::RecordBatch;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::cast::AsArray;
use arrow_array::cast::as_null_array;
use arrow_array::make_array;
use arrow_array::types::ArrowDictionaryKeyType;
use arrow_array::types::ByteArrayType;
use arrow_array::types::ByteViewType;
use arrow_array::types::Date32Type;
use arrow_array::types::Date64Type;
use arrow_array::types::Decimal32Type;
use arrow_array::types::Decimal64Type;
use arrow_array::types::Decimal128Type;
use arrow_array::types::Decimal256Type;
use arrow_array::types::Float16Type;
use arrow_array::types::Float32Type;
use arrow_array::types::Float64Type;
use arrow_array::types::Int8Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::Time32MillisecondType;
use arrow_array::types::Time32SecondType;
use arrow_array::types::Time64MicrosecondType;
use arrow_array::types::Time64NanosecondType;
use arrow_array::types::TimestampMicrosecondType;
use arrow_array::types::TimestampMillisecondType;
use arrow_array::types::TimestampNanosecondType;
use arrow_array::types::TimestampSecondType;
use arrow_array::types::UInt8Type;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use arrow_buffer::ArrowNativeType;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::Buffer as ArrowBuffer;
use arrow_buffer::ScalarBuffer;
use arrow_buffer::buffer::NullBuffer;
use arrow_buffer::buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use itertools::Itertools;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::TemporalArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrow::FromArrowArray;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::extension::datetime::TimeUnit;
use crate::validity::Validity;

impl IntoArray for ArrowBuffer {
    fn into_array(self) -> ArrayRef {
        PrimitiveArray::from_byte_buffer(
            ByteBuffer::from_arrow_buffer(self, Alignment::of::<u8>()),
            PType::U8,
            Validity::NonNullable,
        )
        .into_array()
    }
}

impl IntoArray for BooleanBuffer {
    fn into_array(self) -> ArrayRef {
        BoolArray::new(self.into(), Validity::NonNullable).into_array()
    }
}

impl<T> IntoArray for ScalarBuffer<T>
where
    T: ArrowNativeType + NativePType,
{
    fn into_array(self) -> ArrayRef {
        PrimitiveArray::new(
            Buffer::<T>::from_arrow_scalar_buffer(self),
            Validity::NonNullable,
        )
        .into_array()
    }
}

impl<O> IntoArray for OffsetBuffer<O>
where
    O: IntegerPType + OffsetSizeTrait,
{
    fn into_array(self) -> ArrayRef {
        let primitive = PrimitiveArray::new(
            Buffer::from_arrow_scalar_buffer(self.into_inner()),
            Validity::NonNullable,
        );

        primitive.into_array()
    }
}

macro_rules! impl_from_arrow_primitive {
    ($T:path) => {
        impl FromArrowArray<&ArrowPrimitiveArray<$T>> for ArrayRef {
            fn from_arrow(value: &ArrowPrimitiveArray<$T>, nullable: bool) -> VortexResult<Self> {
                let buffer = Buffer::from_arrow_scalar_buffer(value.values().clone());
                let validity = nulls(value.nulls(), nullable);
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
        }
    };
}

impl_from_arrow_primitive!(Int8Type);
impl_from_arrow_primitive!(Int16Type);
impl_from_arrow_primitive!(Int32Type);
impl_from_arrow_primitive!(Int64Type);
impl_from_arrow_primitive!(UInt8Type);
impl_from_arrow_primitive!(UInt16Type);
impl_from_arrow_primitive!(UInt32Type);
impl_from_arrow_primitive!(UInt64Type);
impl_from_arrow_primitive!(Float16Type);
impl_from_arrow_primitive!(Float32Type);
impl_from_arrow_primitive!(Float64Type);

impl FromArrowArray<&ArrowPrimitiveArray<Decimal32Type>> for ArrayRef {
    fn from_arrow(
        array: &ArrowPrimitiveArray<Decimal32Type>,
        nullable: bool,
    ) -> VortexResult<Self> {
        let decimal_type = DecimalDType::new(array.precision(), array.scale());
        let buffer = Buffer::from_arrow_scalar_buffer(array.values().clone());
        let validity = nulls(array.nulls(), nullable);
        Ok(DecimalArray::new(buffer, decimal_type, validity).into_array())
    }
}

impl FromArrowArray<&ArrowPrimitiveArray<Decimal64Type>> for ArrayRef {
    fn from_arrow(
        array: &ArrowPrimitiveArray<Decimal64Type>,
        nullable: bool,
    ) -> VortexResult<Self> {
        let decimal_type = DecimalDType::new(array.precision(), array.scale());
        let buffer = Buffer::from_arrow_scalar_buffer(array.values().clone());
        let validity = nulls(array.nulls(), nullable);
        Ok(DecimalArray::new(buffer, decimal_type, validity).into_array())
    }
}

impl FromArrowArray<&ArrowPrimitiveArray<Decimal128Type>> for ArrayRef {
    fn from_arrow(
        array: &ArrowPrimitiveArray<Decimal128Type>,
        nullable: bool,
    ) -> VortexResult<Self> {
        let decimal_type = DecimalDType::new(array.precision(), array.scale());
        let buffer = Buffer::from_arrow_scalar_buffer(array.values().clone());
        let validity = nulls(array.nulls(), nullable);
        Ok(DecimalArray::new(buffer, decimal_type, validity).into_array())
    }
}

impl FromArrowArray<&ArrowPrimitiveArray<Decimal256Type>> for ArrayRef {
    fn from_arrow(
        array: &ArrowPrimitiveArray<Decimal256Type>,
        nullable: bool,
    ) -> VortexResult<Self> {
        let decimal_type = DecimalDType::new(array.precision(), array.scale());
        let buffer = Buffer::from_arrow_scalar_buffer(array.values().clone());
        // SAFETY: Our i256 implementation has the same bit-pattern representation of the
        //  arrow_buffer::i256 type. It is safe to treat values held inside the buffer as values
        //  of either type.
        let buffer =
            unsafe { std::mem::transmute::<Buffer<arrow_buffer::i256>, Buffer<i256>>(buffer) };
        let validity = nulls(array.nulls(), nullable);
        Ok(DecimalArray::new(buffer, decimal_type, validity).into_array())
    }
}

macro_rules! impl_from_arrow_temporal {
    ($T:path) => {
        impl FromArrowArray<&ArrowPrimitiveArray<$T>> for ArrayRef {
            fn from_arrow(
                value: &ArrowPrimitiveArray<$T>,
                nullable: bool,
            ) -> vortex_error::VortexResult<Self> {
                Ok(temporal_array(value, nullable))
            }
        }
    };
}

// timestamp
impl_from_arrow_temporal!(TimestampSecondType);
impl_from_arrow_temporal!(TimestampMillisecondType);
impl_from_arrow_temporal!(TimestampMicrosecondType);
impl_from_arrow_temporal!(TimestampNanosecondType);

// time
impl_from_arrow_temporal!(Time32SecondType);
impl_from_arrow_temporal!(Time32MillisecondType);
impl_from_arrow_temporal!(Time64MicrosecondType);
impl_from_arrow_temporal!(Time64NanosecondType);

// date
impl_from_arrow_temporal!(Date32Type);
impl_from_arrow_temporal!(Date64Type);

fn temporal_array<T: ArrowPrimitiveType>(value: &ArrowPrimitiveArray<T>, nullable: bool) -> ArrayRef
where
    T::Native: NativePType,
{
    let arr = PrimitiveArray::new(
        Buffer::from_arrow_scalar_buffer(value.values().clone()),
        nulls(value.nulls(), nullable),
    )
    .into_array();

    match value.data_type() {
        DataType::Timestamp(time_unit, tz) => {
            TemporalArray::new_timestamp(arr, time_unit.into(), tz.clone()).into()
        }
        DataType::Time32(time_unit) => TemporalArray::new_time(arr, time_unit.into()).into(),
        DataType::Time64(time_unit) => TemporalArray::new_time(arr, time_unit.into()).into(),
        DataType::Date32 => TemporalArray::new_date(arr, TimeUnit::Days).into(),
        DataType::Date64 => TemporalArray::new_date(arr, TimeUnit::Milliseconds).into(),
        DataType::Duration(_) => unimplemented!(),
        DataType::Interval(_) => unimplemented!(),
        _ => vortex_panic!("Invalid temporal type: {}", value.data_type()),
    }
}

impl<T: ByteArrayType> FromArrowArray<&GenericByteArray<T>> for ArrayRef
where
    <T as ByteArrayType>::Offset: IntegerPType,
{
    fn from_arrow(value: &GenericByteArray<T>, nullable: bool) -> VortexResult<Self> {
        let dtype = match T::DATA_TYPE {
            DataType::Binary | DataType::LargeBinary => DType::Binary(nullable.into()),
            DataType::Utf8 | DataType::LargeUtf8 => DType::Utf8(nullable.into()),
            dt => vortex_panic!("Invalid data type for ByteArray: {dt}"),
        };
        // SAFETY: Arrow arrays are already validated (valid UTF-8, valid offsets, correct validity).
        Ok(unsafe {
            VarBinArray::new_unchecked(
                value.offsets().clone().into_array(),
                ByteBuffer::from_arrow_buffer(value.values().clone(), Alignment::of::<u8>()),
                dtype,
                nulls(value.nulls(), nullable),
            )
        }
        .into_array())
    }
}

impl<T: ByteViewType> FromArrowArray<&GenericByteViewArray<T>> for ArrayRef {
    fn from_arrow(value: &GenericByteViewArray<T>, nullable: bool) -> VortexResult<Self> {
        let dtype = match T::DATA_TYPE {
            DataType::BinaryView => DType::Binary(nullable.into()),
            DataType::Utf8View => DType::Utf8(nullable.into()),
            dt => vortex_panic!("Invalid data type for ByteViewArray: {dt}"),
        };

        let views_buffer = Buffer::from_byte_buffer(
            Buffer::from_arrow_scalar_buffer(value.views().clone()).into_byte_buffer(),
        );

        // SAFETY: arrow-rs ByteViewArray already checks the same invariants, we inherit those
        //  guarantees by zero-copy constructing from one.
        Ok(unsafe {
            VarBinViewArray::new_unchecked(
                views_buffer,
                Arc::from(
                    value
                        .data_buffers()
                        .iter()
                        .map(|b| ByteBuffer::from_arrow_buffer(b.clone(), Alignment::of::<u8>()))
                        .collect::<Vec<_>>(),
                ),
                dtype,
                nulls(value.nulls(), nullable),
            )
            .into_array()
        })
    }
}

impl FromArrowArray<&ArrowBooleanArray> for ArrayRef {
    fn from_arrow(value: &ArrowBooleanArray, nullable: bool) -> VortexResult<Self> {
        Ok(BoolArray::new(
            value.values().clone().into(),
            nulls(value.nulls(), nullable),
        )
        .into_array())
    }
}

/// Strip out the nulls from this array and return a new array without nulls.
fn remove_nulls(data: arrow_data::ArrayData) -> arrow_data::ArrayData {
    if data.null_count() == 0 {
        // No nulls to remove, return the array as is
        return data;
    }

    let children = match data.data_type() {
        DataType::Struct(fields) => Some(
            fields
                .iter()
                .zip(data.child_data().iter())
                .map(|(field, child_data)| {
                    if field.is_nullable() {
                        child_data.clone()
                    } else {
                        remove_nulls(child_data.clone())
                    }
                })
                .collect_vec(),
        ),
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
            if !f.is_nullable() =>
        {
            // All list types only have one child
            assert_eq!(
                data.child_data().len(),
                1,
                "List types should have one child"
            );
            Some(vec![remove_nulls(data.child_data()[0].clone())])
        }
        _ => None,
    };

    let mut builder = data.into_builder().nulls(None);
    if let Some(children) = children {
        builder = builder.child_data(children);
    }
    builder
        .build()
        .vortex_expect("reconstructing array without nulls")
}

impl FromArrowArray<&ArrowStructArray> for ArrayRef {
    fn from_arrow(value: &ArrowStructArray, nullable: bool) -> VortexResult<Self> {
        Ok(StructArray::try_new(
            value.column_names().iter().copied().collect(),
            value
                .columns()
                .iter()
                .zip(value.fields())
                .map(|(c, field)| {
                    // Arrow pushes down nulls, even into non-nullable fields. So we strip them
                    // out here because Vortex is a little more strict.
                    if c.null_count() > 0 && !field.is_nullable() {
                        let stripped = make_array(remove_nulls(c.into_data()));
                        Self::from_arrow(stripped.as_ref(), false)
                    } else {
                        Self::from_arrow(c.as_ref(), field.is_nullable())
                    }
                })
                .collect::<VortexResult<Vec<_>>>()?,
            value.len(),
            nulls(value.nulls(), nullable),
        )?
        .into_array())
    }
}

impl<O: IntegerPType + OffsetSizeTrait> FromArrowArray<&GenericListArray<O>> for ArrayRef {
    fn from_arrow(value: &GenericListArray<O>, nullable: bool) -> VortexResult<Self> {
        // Extract the validity of the underlying element array.
        let elements_are_nullable = match value.data_type() {
            DataType::List(field) => field.is_nullable(),
            DataType::LargeList(field) => field.is_nullable(),
            dt => vortex_panic!("Invalid data type for ListArray: {dt}"),
        };

        let elements = Self::from_arrow(value.values().as_ref(), elements_are_nullable)?;

        // `offsets` are always non-nullable.
        let offsets = value.offsets().clone().into_array();
        let nulls = nulls(value.nulls(), nullable);

        Ok(ListArray::try_new(elements, offsets, nulls)?.into_array())
    }
}

impl<O: OffsetSizeTrait + NativePType> FromArrowArray<&GenericListViewArray<O>> for ArrayRef {
    fn from_arrow(array: &GenericListViewArray<O>, nullable: bool) -> VortexResult<Self> {
        // Extract the validity of the underlying element array.
        let elements_are_nullable = match array.data_type() {
            DataType::ListView(field) => field.is_nullable(),
            DataType::LargeListView(field) => field.is_nullable(),
            dt => vortex_panic!("Invalid data type for ListViewArray: {dt}"),
        };

        let elements = Self::from_arrow(array.values().as_ref(), elements_are_nullable)?;

        // `offsets` and `sizes` are always non-nullable.
        let offsets = array.offsets().clone().into_array();
        let sizes = array.sizes().clone().into_array();
        let nulls = nulls(array.nulls(), nullable);

        Ok(ListViewArray::try_new(elements, offsets, sizes, nulls)?.into_array())
    }
}

impl FromArrowArray<&ArrowFixedSizeListArray> for ArrayRef {
    fn from_arrow(array: &ArrowFixedSizeListArray, nullable: bool) -> VortexResult<Self> {
        let DataType::FixedSizeList(field, list_size) = array.data_type() else {
            vortex_panic!("Invalid data type for ListArray: {}", array.data_type());
        };

        Ok(FixedSizeListArray::try_new(
            Self::from_arrow(array.values().as_ref(), field.is_nullable())?,
            *list_size as u32,
            nulls(array.nulls(), nullable),
            array.len(),
        )?
        .into_array())
    }
}

impl FromArrowArray<&ArrowNullArray> for ArrayRef {
    fn from_arrow(value: &ArrowNullArray, nullable: bool) -> VortexResult<Self> {
        assert!(nullable);
        Ok(NullArray::new(value.len()).into_array())
    }
}

impl<K: ArrowDictionaryKeyType> FromArrowArray<&DictionaryArray<K>> for DictArray {
    fn from_arrow(array: &DictionaryArray<K>, nullable: bool) -> VortexResult<Self> {
        let keys = AnyDictionaryArray::keys(array);
        let keys = ArrayRef::from_arrow(keys, keys.is_nullable())?;
        let values = ArrayRef::from_arrow(array.values().as_ref(), nullable)?;
        // SAFETY: we assume that Arrow has checked the invariants on construction.
        Ok(unsafe { DictArray::new_unchecked(keys, values) })
    }
}

fn nulls(nulls: Option<&NullBuffer>, nullable: bool) -> Validity {
    if nullable {
        nulls
            .map(|nulls| {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::from(BitBuffer::from(nulls.inner().clone()))
                }
            })
            .unwrap_or_else(|| Validity::AllValid)
    } else {
        assert!(nulls.map(|x| x.null_count() == 0).unwrap_or(true));
        Validity::NonNullable
    }
}

impl FromArrowArray<&dyn ArrowArray> for ArrayRef {
    fn from_arrow(array: &dyn ArrowArray, nullable: bool) -> VortexResult<Self> {
        match array.data_type() {
            DataType::Boolean => Self::from_arrow(array.as_boolean(), nullable),
            DataType::UInt8 => Self::from_arrow(array.as_primitive::<UInt8Type>(), nullable),
            DataType::UInt16 => Self::from_arrow(array.as_primitive::<UInt16Type>(), nullable),
            DataType::UInt32 => Self::from_arrow(array.as_primitive::<UInt32Type>(), nullable),
            DataType::UInt64 => Self::from_arrow(array.as_primitive::<UInt64Type>(), nullable),
            DataType::Int8 => Self::from_arrow(array.as_primitive::<Int8Type>(), nullable),
            DataType::Int16 => Self::from_arrow(array.as_primitive::<Int16Type>(), nullable),
            DataType::Int32 => Self::from_arrow(array.as_primitive::<Int32Type>(), nullable),
            DataType::Int64 => Self::from_arrow(array.as_primitive::<Int64Type>(), nullable),
            DataType::Float16 => Self::from_arrow(array.as_primitive::<Float16Type>(), nullable),
            DataType::Float32 => Self::from_arrow(array.as_primitive::<Float32Type>(), nullable),
            DataType::Float64 => Self::from_arrow(array.as_primitive::<Float64Type>(), nullable),
            DataType::Utf8 => Self::from_arrow(array.as_string::<i32>(), nullable),
            DataType::LargeUtf8 => Self::from_arrow(array.as_string::<i64>(), nullable),
            DataType::Binary => Self::from_arrow(array.as_binary::<i32>(), nullable),
            DataType::LargeBinary => Self::from_arrow(array.as_binary::<i64>(), nullable),
            DataType::BinaryView => Self::from_arrow(array.as_binary_view(), nullable),
            DataType::Utf8View => Self::from_arrow(array.as_string_view(), nullable),
            DataType::Struct(_) => Self::from_arrow(array.as_struct(), nullable),
            DataType::List(_) => Self::from_arrow(array.as_list::<i32>(), nullable),
            DataType::LargeList(_) => Self::from_arrow(array.as_list::<i64>(), nullable),
            DataType::ListView(_) => Self::from_arrow(array.as_list_view::<i32>(), nullable),
            DataType::LargeListView(_) => Self::from_arrow(array.as_list_view::<i64>(), nullable),
            DataType::FixedSizeList(..) => Self::from_arrow(array.as_fixed_size_list(), nullable),
            DataType::Null => Self::from_arrow(as_null_array(array), nullable),
            DataType::Timestamp(u, _) => match u {
                ArrowTimeUnit::Second => {
                    Self::from_arrow(array.as_primitive::<TimestampSecondType>(), nullable)
                }
                ArrowTimeUnit::Millisecond => {
                    Self::from_arrow(array.as_primitive::<TimestampMillisecondType>(), nullable)
                }
                ArrowTimeUnit::Microsecond => {
                    Self::from_arrow(array.as_primitive::<TimestampMicrosecondType>(), nullable)
                }
                ArrowTimeUnit::Nanosecond => {
                    Self::from_arrow(array.as_primitive::<TimestampNanosecondType>(), nullable)
                }
            },
            DataType::Date32 => Self::from_arrow(array.as_primitive::<Date32Type>(), nullable),
            DataType::Date64 => Self::from_arrow(array.as_primitive::<Date64Type>(), nullable),
            DataType::Time32(u) => match u {
                ArrowTimeUnit::Second => {
                    Self::from_arrow(array.as_primitive::<Time32SecondType>(), nullable)
                }
                ArrowTimeUnit::Millisecond => {
                    Self::from_arrow(array.as_primitive::<Time32MillisecondType>(), nullable)
                }
                ArrowTimeUnit::Microsecond | ArrowTimeUnit::Nanosecond => unreachable!(),
            },
            DataType::Time64(u) => match u {
                ArrowTimeUnit::Microsecond => {
                    Self::from_arrow(array.as_primitive::<Time64MicrosecondType>(), nullable)
                }
                ArrowTimeUnit::Nanosecond => {
                    Self::from_arrow(array.as_primitive::<Time64NanosecondType>(), nullable)
                }
                ArrowTimeUnit::Second | ArrowTimeUnit::Millisecond => unreachable!(),
            },
            DataType::Decimal32(..) => {
                Self::from_arrow(array.as_primitive::<Decimal32Type>(), nullable)
            }
            DataType::Decimal64(..) => {
                Self::from_arrow(array.as_primitive::<Decimal64Type>(), nullable)
            }
            DataType::Decimal128(..) => {
                Self::from_arrow(array.as_primitive::<Decimal128Type>(), nullable)
            }
            DataType::Decimal256(..) => {
                Self::from_arrow(array.as_primitive::<Decimal256Type>(), nullable)
            }
            DataType::Dictionary(key_type, _) => match key_type.as_ref() {
                DataType::Int8 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<Int8Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::Int16 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<Int16Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::Int32 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<Int32Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::Int64 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<Int64Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::UInt8 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<UInt8Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::UInt16 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<UInt16Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::UInt32 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<UInt32Type>(),
                    nullable,
                )?
                .into_array()),
                DataType::UInt64 => Ok(DictArray::from_arrow(
                    array.as_dictionary::<UInt64Type>(),
                    nullable,
                )?
                .into_array()),
                key_dt => vortex_bail!("Unsupported dictionary key type: {key_dt}"),
            },
            dt => vortex_bail!("Array encoding not implemented for Arrow data type {dt}"),
        }
    }
}

impl FromArrowArray<RecordBatch> for ArrayRef {
    fn from_arrow(array: RecordBatch, nullable: bool) -> VortexResult<Self> {
        ArrayRef::from_arrow(&arrow_array::StructArray::from(array), nullable)
    }
}

impl FromArrowArray<&RecordBatch> for ArrayRef {
    fn from_arrow(array: &RecordBatch, nullable: bool) -> VortexResult<Self> {
        Self::from_arrow(array.clone(), nullable)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array as ArrowArray;
    use arrow_array::BinaryArray;
    use arrow_array::BooleanArray;
    use arrow_array::Date32Array;
    use arrow_array::Date64Array;
    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use arrow_array::Float64Array;
    use arrow_array::GenericListViewArray;
    use arrow_array::Int8Array;
    use arrow_array::Int16Array;
    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_array::LargeBinaryArray;
    use arrow_array::LargeStringArray;
    use arrow_array::NullArray;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_array::Time32MillisecondArray;
    use arrow_array::Time32SecondArray;
    use arrow_array::Time64MicrosecondArray;
    use arrow_array::Time64NanosecondArray;
    use arrow_array::TimestampMicrosecondArray;
    use arrow_array::TimestampMillisecondArray;
    use arrow_array::TimestampNanosecondArray;
    use arrow_array::TimestampSecondArray;
    use arrow_array::UInt8Array;
    use arrow_array::UInt16Array;
    use arrow_array::UInt32Array;
    use arrow_array::UInt64Array;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_array::builder::Decimal128Builder;
    use arrow_array::builder::Decimal256Builder;
    use arrow_array::builder::Int32Builder;
    use arrow_array::builder::LargeListBuilder;
    use arrow_array::builder::ListBuilder;
    use arrow_array::builder::StringViewBuilder;
    use arrow_array::new_null_array;
    use arrow_array::types::ArrowPrimitiveType;
    use arrow_array::types::Float16Type;
    use arrow_buffer::BooleanBuffer;
    use arrow_buffer::Buffer as ArrowBuffer;
    use arrow_buffer::OffsetBuffer;
    use arrow_buffer::ScalarBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use arrow_schema::Schema;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::Decimal;
    use crate::arrays::FixedSizeList;
    use crate::arrays::List;
    use crate::arrays::ListView;
    use crate::arrays::Primitive;
    use crate::arrays::Struct;
    use crate::arrays::VarBin;
    use crate::arrays::VarBinView;
    use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
    use crate::arrays::list::ListArrayExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::arrays::struct_::StructArrayExt;
    use crate::arrow::FromArrowArray as _;
    use crate::arrow::convert::TemporalArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    // Test primitive array conversions
    #[test]
    fn test_int8_array_conversion() {
        let arrow_array = Int8Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Int8Array::from(vec![1, 2, 3, 4]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I8 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::I8);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I8);
    }

    #[test]
    fn test_int16_array_conversion() {
        let arrow_array = Int16Array::from(vec![Some(100), None, Some(300), Some(400)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Int16Array::from(vec![100, 200, 300, 400]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I16 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::I16);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I16);
    }

    #[test]
    fn test_int32_array_conversion() {
        let arrow_array = Int32Array::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Int32Array::from(vec![1000, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I32 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::I32);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I32);
    }

    #[test]
    fn test_int64_array_conversion() {
        let arrow_array = Int64Array::from(vec![Some(10000), None, Some(30000), Some(40000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Int64Array::from(vec![10000_i64, 20000, 30000, 40000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I64 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::I64);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I64);
    }

    #[test]
    fn test_uint8_array_conversion() {
        let arrow_array = UInt8Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = UInt8Array::from(vec![1_u8, 2, 3, 4]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U8 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::U8);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U8);
    }

    #[test]
    fn test_uint16_array_conversion() {
        let arrow_array = UInt16Array::from(vec![Some(100), None, Some(300), Some(400)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = UInt16Array::from(vec![100_u16, 200, 300, 400]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U16 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::U16);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U16);
    }

    #[test]
    fn test_uint32_array_conversion() {
        let arrow_array = UInt32Array::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = UInt32Array::from(vec![1000_u32, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U32 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::U32);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U32);
    }

    #[test]
    fn test_uint64_array_conversion() {
        let arrow_array = UInt64Array::from(vec![Some(10000), None, Some(30000), Some(40000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = UInt64Array::from(vec![10000_u64, 20000, 30000, 40000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U64 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::U64);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U64);
    }

    #[test]
    fn test_float16_array_conversion() {
        let values = vec![
            Some(<Float16Type as ArrowPrimitiveType>::Native::from_f32(1.5)),
            None,
            Some(<Float16Type as ArrowPrimitiveType>::Native::from_f32(3.5)),
        ];
        let arrow_array = arrow_array::PrimitiveArray::<Float16Type>::from(values);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let non_null_values = vec![
            <Float16Type as ArrowPrimitiveType>::Native::from_f32(1.5),
            <Float16Type as ArrowPrimitiveType>::Native::from_f32(2.5),
        ];
        let arrow_array_non_null =
            arrow_array::PrimitiveArray::<Float16Type>::from(non_null_values);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 3);
        assert_eq!(vortex_array_non_null.len(), 2);

        // Verify metadata - should be PrimitiveArray with F16 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::F16);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::F16);
    }

    #[test]
    fn test_float32_array_conversion() {
        let arrow_array = Float32Array::from(vec![Some(1.5), None, Some(3.5), Some(4.5)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Float32Array::from(vec![1.5_f32, 2.5, 3.5, 4.5]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with F32 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::F32);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::F32);
    }

    #[test]
    fn test_float64_array_conversion() {
        let arrow_array = Float64Array::from(vec![Some(1.5), None, Some(3.5), Some(4.5)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Float64Array::from(vec![1.5_f64, 2.5, 3.5, 4.5]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with F64 ptype
        let primitive_array = vortex_array.as_::<Primitive>();
        assert_eq!(primitive_array.ptype(), PType::F64);

        let primitive_array_non_null = vortex_array_non_null.as_::<Primitive>();
        assert_eq!(primitive_array_non_null.ptype(), PType::F64);
    }

    // Test decimal array conversions
    #[test]
    fn test_decimal128_array_conversion() {
        let mut builder = Decimal128Builder::with_capacity(4);
        builder.append_value(12345);
        builder.append_null();
        builder.append_value(67890);
        builder.append_value(11111);
        let decimal_array = builder.finish().with_precision_and_scale(10, 2).unwrap();

        let vortex_array = ArrayRef::from_arrow(&decimal_array, true).unwrap();
        assert_eq!(vortex_array.len(), 4);

        let mut builder_non_null = Decimal128Builder::with_capacity(3);
        builder_non_null.append_value(12345);
        builder_non_null.append_value(67890);
        builder_non_null.append_value(11111);
        let decimal_array_non_null = builder_non_null
            .finish()
            .with_precision_and_scale(10, 2)
            .unwrap();

        let vortex_array_non_null = ArrayRef::from_arrow(&decimal_array_non_null, false).unwrap();
        assert_eq!(vortex_array_non_null.len(), 3);

        // Verify metadata - should be DecimalArray with correct precision and scale
        let decimal_vortex_array = vortex_array.as_::<Decimal>();
        assert_eq!(decimal_vortex_array.decimal_dtype().precision(), 10);
        assert_eq!(decimal_vortex_array.decimal_dtype().scale(), 2);

        let decimal_vortex_array_non_null = vortex_array_non_null.as_::<Decimal>();
        assert_eq!(
            decimal_vortex_array_non_null.decimal_dtype().precision(),
            10
        );
        assert_eq!(decimal_vortex_array_non_null.decimal_dtype().scale(), 2);
    }

    #[test]
    fn test_decimal256_array_conversion() {
        let mut builder = Decimal256Builder::with_capacity(4);
        builder.append_value(arrow_buffer::i256::from_i128(12345));
        builder.append_null();
        builder.append_value(arrow_buffer::i256::from_i128(67890));
        builder.append_value(arrow_buffer::i256::from_i128(11111));
        let decimal_array = builder.finish().with_precision_and_scale(38, 10).unwrap();

        let vortex_array = ArrayRef::from_arrow(&decimal_array, true).unwrap();
        assert_eq!(vortex_array.len(), 4);

        let mut builder_non_null = Decimal256Builder::with_capacity(3);
        builder_non_null.append_value(arrow_buffer::i256::from_i128(12345));
        builder_non_null.append_value(arrow_buffer::i256::from_i128(67890));
        builder_non_null.append_value(arrow_buffer::i256::from_i128(11111));
        let decimal_array_non_null = builder_non_null
            .finish()
            .with_precision_and_scale(38, 10)
            .unwrap();

        let vortex_array_non_null = ArrayRef::from_arrow(&decimal_array_non_null, false).unwrap();
        assert_eq!(vortex_array_non_null.len(), 3);

        // Verify metadata - should be DecimalArray with correct precision and scale
        let decimal_vortex_array = vortex_array.as_::<Decimal>();
        assert_eq!(decimal_vortex_array.decimal_dtype().precision(), 38);
        assert_eq!(decimal_vortex_array.decimal_dtype().scale(), 10);

        let decimal_vortex_array_non_null = vortex_array_non_null.as_::<Decimal>();
        assert_eq!(
            decimal_vortex_array_non_null.decimal_dtype().precision(),
            38
        );
        assert_eq!(decimal_vortex_array_non_null.decimal_dtype().scale(), 10);
    }

    // Test temporal array conversions
    #[test]
    fn test_timestamp_second_array_conversion() {
        let arrow_array =
            TimestampSecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = TimestampSecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be TemporalArray with Second time unit
        let temporal_array = TemporalArray::try_from(vortex_array).unwrap();
        assert_eq!(
            temporal_array.temporal_metadata().time_unit(),
            TimeUnit::Seconds
        );

        let temporal_array_non_null = TemporalArray::try_from(vortex_array_non_null).unwrap();
        assert_eq!(
            temporal_array_non_null.temporal_metadata().time_unit(),
            TimeUnit::Seconds
        );
    }

    #[test]
    fn test_timestamp_millisecond_array_conversion() {
        let arrow_array =
            TimestampMillisecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null =
            TimestampMillisecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_timestamp_microsecond_array_conversion() {
        let arrow_array =
            TimestampMicrosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null =
            TimestampMicrosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_timestamp_timezone_microsecond_array_conversion() {
        let arrow_array =
            TimestampMicrosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)])
                .with_timezone("UTC");
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null =
            TimestampMicrosecondArray::from(vec![1000_i64, 2000, 3000, 4000]).with_timezone("UTC");
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Extension(
                Timestamp::new_with_tz(
                    TimeUnit::Microseconds,
                    Some("UTC".into()),
                    Nullability::Nullable
                )
                .erased()
            ),
        );
        assert_eq!(vortex_array_non_null.len(), 4);
        assert_eq!(
            vortex_array_non_null.dtype(),
            &DType::Extension(
                Timestamp::new_with_tz(
                    TimeUnit::Microseconds,
                    Some("UTC".into()),
                    Nullability::NonNullable
                )
                .erased()
            )
        );
    }

    #[test]
    fn test_timestamp_nanosecond_array_conversion() {
        let arrow_array =
            TimestampNanosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = TimestampNanosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_time32_second_array_conversion() {
        let arrow_array = Time32SecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Time32SecondArray::from(vec![1000_i32, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be TemporalArray with Second time unit
        let temporal_array = TemporalArray::try_from(vortex_array).unwrap();
        assert_eq!(
            temporal_array.temporal_metadata().time_unit(),
            TimeUnit::Seconds
        );

        let temporal_array_non_null = TemporalArray::try_from(vortex_array_non_null).unwrap();
        assert_eq!(
            temporal_array_non_null.temporal_metadata().time_unit(),
            TimeUnit::Seconds
        );
    }

    #[test]
    fn test_time32_millisecond_array_conversion() {
        let arrow_array =
            Time32MillisecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Time32MillisecondArray::from(vec![1000_i32, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_time64_microsecond_array_conversion() {
        let arrow_array =
            Time64MicrosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Time64MicrosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_time64_nanosecond_array_conversion() {
        let arrow_array =
            Time64NanosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Time64NanosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_date32_array_conversion() {
        let arrow_array = Date32Array::from(vec![Some(18000), None, Some(18002), Some(18003)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Date32Array::from(vec![18000_i32, 18001, 18002, 18003]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_date64_array_conversion() {
        let arrow_array = Date64Array::from(vec![
            Some(1555200000000),
            None,
            Some(1555286400000),
            Some(1555372800000),
        ]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = Date64Array::from(vec![
            1555200000000_i64,
            1555213600000,
            1555286400000,
            1555372800000,
        ]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    // Test string/binary array conversions
    #[test]
    fn test_utf8_array_conversion() {
        let arrow_array = StringArray::from(vec![Some("hello"), None, Some("world"), Some("test")]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = StringArray::from(vec!["hello", "world", "test", "vortex"]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be VarBinArray with Utf8 dtype
        let varbin_array = vortex_array.as_::<VarBin>();
        assert_eq!(varbin_array.dtype(), &DType::Utf8(true.into()));

        let varbin_array_non_null = vortex_array_non_null.as_::<VarBin>();
        assert_eq!(varbin_array_non_null.dtype(), &DType::Utf8(false.into()));
    }

    #[test]
    fn test_large_utf8_array_conversion() {
        let arrow_array =
            LargeStringArray::from(vec![Some("hello"), None, Some("world"), Some("test")]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = LargeStringArray::from(vec!["hello", "world", "test", "vortex"]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_binary_array_conversion() {
        let arrow_array = BinaryArray::from(vec![
            Some("hello".as_bytes()),
            None,
            Some("world".as_bytes()),
            Some("test".as_bytes()),
        ]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = BinaryArray::from(vec![
            "hello".as_bytes(),
            "world".as_bytes(),
            "test".as_bytes(),
            "vortex".as_bytes(),
        ]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_large_binary_array_conversion() {
        let arrow_array = LargeBinaryArray::from(vec![
            Some("hello".as_bytes()),
            None,
            Some("world".as_bytes()),
            Some("test".as_bytes()),
        ]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = LargeBinaryArray::from(vec![
            "hello".as_bytes(),
            "world".as_bytes(),
            "test".as_bytes(),
            "vortex".as_bytes(),
        ]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_utf8_view_array_conversion() {
        let mut builder = StringViewBuilder::new();
        builder.append_value("hello");
        builder.append_null();
        builder.append_value("world");
        builder.append_value("test");
        let arrow_array = builder.finish();
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let mut builder_non_null = StringViewBuilder::new();
        builder_non_null.append_value("hello");
        builder_non_null.append_value("world");
        builder_non_null.append_value("test");
        builder_non_null.append_value("vortex");
        let arrow_array_non_null = builder_non_null.finish();
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be VarBinViewArray with correct buffer count and dtype
        let varbin_view_array = vortex_array.as_::<VarBinView>();
        assert_eq!(
            varbin_view_array.data_buffers().len(),
            arrow_array.data_buffers().len()
        );
        assert_eq!(varbin_view_array.dtype(), &DType::Utf8(true.into()));

        let varbin_view_array_non_null = vortex_array_non_null.as_::<VarBinView>();
        assert_eq!(
            varbin_view_array_non_null.data_buffers().len(),
            arrow_array_non_null.data_buffers().len()
        );
        assert_eq!(
            varbin_view_array_non_null.dtype(),
            &DType::Utf8(false.into())
        );
    }

    #[test]
    fn test_binary_view_array_conversion() {
        let mut builder = BinaryViewBuilder::new();
        builder.append_value(b"hello");
        builder.append_null();
        builder.append_value(b"world");
        builder.append_value(b"test");
        let arrow_array = builder.finish();
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let mut builder_non_null = BinaryViewBuilder::new();
        builder_non_null.append_value(b"hello");
        builder_non_null.append_value(b"world");
        builder_non_null.append_value(b"test");
        builder_non_null.append_value(b"vortex");
        let arrow_array_non_null = builder_non_null.finish();
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be VarBinViewArray with correct buffer count and dtype
        let varbin_view_array = vortex_array.as_::<VarBinView>();
        assert_eq!(
            varbin_view_array.data_buffers().len(),
            arrow_array.data_buffers().len()
        );
        assert_eq!(varbin_view_array.dtype(), &DType::Binary(true.into()));

        let varbin_view_array_non_null = vortex_array_non_null.as_::<VarBinView>();
        assert_eq!(
            varbin_view_array_non_null.data_buffers().len(),
            arrow_array_non_null.data_buffers().len()
        );
        assert_eq!(
            varbin_view_array_non_null.dtype(),
            &DType::Binary(false.into())
        );
    }

    // Test boolean array conversions
    #[test]
    fn test_boolean_array_conversion() {
        let arrow_array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();

        let arrow_array_non_null = BooleanArray::from(vec![true, false, true, false]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    // Test struct array conversions
    #[test]
    fn test_struct_array_conversion() {
        let fields = vec![
            Field::new("field1", DataType::Int32, true),
            Field::new("field2", DataType::Utf8, false),
        ];
        let schema = Fields::from(fields);

        let field1_data = Int32Array::from(vec![Some(1), None, Some(3)]);
        let field2_data = StringArray::from(vec!["a", "b", "c"]);

        let arrow_array = StructArray::new(
            schema.clone(),
            vec![Arc::new(field1_data), Arc::new(field2_data)],
            None,
        );

        let vortex_array = ArrayRef::from_arrow(&arrow_array, false).unwrap();
        assert_eq!(vortex_array.len(), 3);

        // Verify metadata - should be StructArray with correct field names
        let struct_vortex_array = vortex_array.as_::<Struct>();
        assert_eq!(struct_vortex_array.names().len(), 2);
        assert_eq!(struct_vortex_array.names()[0], "field1");
        assert_eq!(struct_vortex_array.names()[1], "field2");

        // Test nullable struct
        let nullable_array = StructArray::new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
            Some(arrow_buffer::NullBuffer::new(BooleanBuffer::from(vec![
                true, false, true,
            ]))),
        );

        let vortex_nullable_array = ArrayRef::from_arrow(&nullable_array, true).unwrap();
        assert_eq!(vortex_nullable_array.len(), 3);

        // Verify metadata for nullable struct
        let struct_vortex_nullable_array = vortex_nullable_array.as_::<Struct>();
        assert_eq!(struct_vortex_nullable_array.names().len(), 2);
        assert_eq!(struct_vortex_nullable_array.names()[0], "field1");
        assert_eq!(struct_vortex_nullable_array.names()[1], "field2");
    }

    // Test list array conversions
    #[test]
    fn test_list_array_conversion() {
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.append_value([Some(1), None, Some(3)]);
        builder.append_null();
        builder.append_value([Some(4), Some(5)]);
        let arrow_array = builder.finish();

        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();
        assert_eq!(vortex_array.len(), 3);

        // Verify metadata - should be ListArray with correct offsets
        let list_vortex_array = vortex_array.as_::<List>();
        let offsets_array = list_vortex_array.offsets().as_::<Primitive>();
        assert_eq!(offsets_array.len(), 4); // n+1 offsets for n lists
        assert_eq!(offsets_array.ptype(), PType::I32);

        // Test non-nullable list
        let mut builder_non_null = ListBuilder::new(Int32Builder::new());
        builder_non_null.append_value([Some(1), None, Some(3)]);
        builder_non_null.append_value([Some(4), Some(5)]);
        let arrow_array_non_null = builder_non_null.finish();

        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();
        assert_eq!(vortex_array_non_null.len(), 2);

        // Verify metadata for non-nullable list
        let list_vortex_array_non_null = vortex_array_non_null.as_::<List>();
        let offsets_array_non_null = list_vortex_array_non_null.offsets().as_::<Primitive>();
        assert_eq!(offsets_array_non_null.len(), 3); // n+1 offsets for n lists
        assert_eq!(offsets_array_non_null.ptype(), PType::I32);
    }

    #[test]
    fn test_large_list_array_conversion() {
        let mut builder = LargeListBuilder::new(Int32Builder::new());
        builder.append_value([Some(1), None, Some(3)]);
        builder.append_null();
        builder.append_value([Some(4), Some(5)]);
        let arrow_array = builder.finish();

        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();
        assert_eq!(vortex_array.len(), 3);

        // Verify metadata - should be ListArray with correct offsets (I64 for large lists)
        let list_vortex_array = vortex_array.as_::<List>();
        let offsets_array = list_vortex_array.offsets().as_::<Primitive>();
        assert_eq!(offsets_array.len(), 4); // n+1 offsets for n lists
        assert_eq!(offsets_array.ptype(), PType::I64); // Large lists use I64 offsets

        // Test non-nullable large list
        let mut builder_non_null = LargeListBuilder::new(Int32Builder::new());
        builder_non_null.append_value([Some(1), None, Some(3)]);
        builder_non_null.append_value([Some(4), Some(5)]);
        let arrow_array_non_null = builder_non_null.finish();

        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false).unwrap();
        assert_eq!(vortex_array_non_null.len(), 2);

        // Verify metadata for non-nullable large list
        let list_vortex_array_non_null = vortex_array_non_null.as_::<List>();
        let offsets_array_non_null = list_vortex_array_non_null.offsets().as_::<Primitive>();
        assert_eq!(offsets_array_non_null.len(), 3); // n+1 offsets for n lists
        assert_eq!(offsets_array_non_null.ptype(), PType::I64); // Large lists use I64 offsets
    }

    #[test]
    fn test_fixed_size_list_array_conversion() {
        // Create elements for the fixed-size lists
        let values = Int32Array::from(vec![
            Some(1),
            Some(2),
            Some(3), // First list
            Some(4),
            None,
            Some(6), // Second list (with null element)
            Some(7),
            Some(8),
            Some(9), // Third list
            Some(10),
            Some(11),
            Some(12), // Fourth list
        ]);

        // Create a FixedSizeListArray with list_size=3
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let arrow_array =
            ArrowFixedSizeListArray::try_new(Arc::clone(&field), 3, Arc::new(values), None)
                .unwrap();
        let vortex_array = ArrayRef::from_arrow(&arrow_array, false).unwrap();

        assert_eq!(vortex_array.len(), 4);

        // Verify metadata - should be FixedSizeListArray with correct list size
        let fsl_vortex_array = vortex_array.as_::<FixedSizeList>();
        assert_eq!(fsl_vortex_array.list_size(), 3);
        assert_eq!(fsl_vortex_array.elements().len(), 12); // 4 lists * 3 elements

        // Test nullable fixed-size list
        let values_nullable = Int32Array::from(vec![
            Some(1),
            Some(2),
            Some(3), // First list
            Some(4),
            None,
            Some(6), // Second list (will be null)
            Some(7),
            Some(8),
            Some(9), // Third list
        ]);

        // Create nulls buffer - second list is null
        let null_buffer =
            arrow_buffer::NullBuffer::new(BooleanBuffer::from(vec![true, false, true]));

        let arrow_array_nullable = ArrowFixedSizeListArray::try_new(
            field,
            3,
            Arc::new(values_nullable),
            Some(null_buffer),
        )
        .unwrap();
        let vortex_array_nullable = ArrayRef::from_arrow(&arrow_array_nullable, true).unwrap();

        assert_eq!(vortex_array_nullable.len(), 3);

        // Verify metadata for nullable array
        let fsl_vortex_array_nullable = vortex_array_nullable.as_::<FixedSizeList>();
        assert_eq!(fsl_vortex_array_nullable.list_size(), 3);
        assert_eq!(fsl_vortex_array_nullable.elements().len(), 9); // 3 lists * 3 elements
    }

    #[test]
    fn test_list_view_array_conversion() {
        // Create values array for the lists
        let values = Int32Array::from(vec![
            Some(1),
            Some(2),
            Some(3), // First list [1, 2, 3]
            Some(4),
            Some(5), // Second list [4, 5]
            Some(6), // Third list [6]
            Some(7),
            Some(8),
            Some(9),
            Some(10), // Fourth list [7, 8, 9, 10]
        ]);

        // Create offsets and sizes for ListView
        let offsets = ScalarBuffer::from(vec![0i32, 3, 5, 6]);
        let sizes = ScalarBuffer::from(vec![3i32, 2, 1, 4]);

        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let arrow_array = GenericListViewArray::try_new(
            Arc::clone(&field),
            offsets.clone(),
            sizes.clone(),
            Arc::new(values.clone()),
            None,
        )
        .unwrap();

        let vortex_array = ArrayRef::from_arrow(&arrow_array, false).unwrap();
        assert_eq!(vortex_array.len(), 4);

        // Verify metadata - should be ListViewArray with correct offsets and sizes
        let list_view_vortex_array = vortex_array.as_::<ListView>();
        let offsets_array = list_view_vortex_array.offsets().as_::<Primitive>();
        let sizes_array = list_view_vortex_array.sizes().as_::<Primitive>();

        assert_eq!(offsets_array.len(), 4);
        assert_eq!(offsets_array.ptype(), PType::I32);
        assert_eq!(sizes_array.len(), 4);
        assert_eq!(sizes_array.ptype(), PType::I32);

        // Test nullable ListView
        let null_buffer =
            arrow_buffer::NullBuffer::new(BooleanBuffer::from(vec![true, false, true, true]));

        let arrow_array_nullable = GenericListViewArray::try_new(
            Arc::clone(&field),
            offsets,
            sizes,
            Arc::new(values.clone()),
            Some(null_buffer),
        )
        .unwrap();

        let vortex_array_nullable = ArrayRef::from_arrow(&arrow_array_nullable, true).unwrap();
        assert_eq!(vortex_array_nullable.len(), 4);

        // Test LargeListView (i64 offsets and sizes)
        let large_offsets = ScalarBuffer::from(vec![0i64, 3, 5, 6]);
        let large_sizes = ScalarBuffer::from(vec![3i64, 2, 1, 4]);

        let large_arrow_array = GenericListViewArray::try_new(
            field,
            large_offsets,
            large_sizes,
            Arc::new(values),
            None,
        )
        .unwrap();

        let large_vortex_array = ArrayRef::from_arrow(&large_arrow_array, false).unwrap();
        assert_eq!(large_vortex_array.len(), 4);

        // Verify metadata for large ListView
        let large_list_view_vortex_array = large_vortex_array.as_::<ListView>();
        let large_offsets_array = large_list_view_vortex_array.offsets().as_::<Primitive>();
        let large_sizes_array = large_list_view_vortex_array.sizes().as_::<Primitive>();

        assert_eq!(large_offsets_array.len(), 4);
        assert_eq!(large_offsets_array.ptype(), PType::I64); // Large ListView uses I64 offsets
        assert_eq!(large_sizes_array.len(), 4);
        assert_eq!(large_sizes_array.ptype(), PType::I64); // Large ListView uses I64 sizes
    }

    // Test null array conversions
    #[test]
    fn test_null_array_conversion() {
        let arrow_array = NullArray::new(5);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true).unwrap();
        assert_eq!(vortex_array.len(), 5);
    }

    // Test buffer conversions
    #[test]
    fn test_arrow_buffer_conversion() {
        let data = vec![1u8, 2, 3, 4, 5];
        let arrow_buffer = ArrowBuffer::from_vec(data);
        let vortex_array = arrow_buffer.into_array();
        assert_eq!(vortex_array.len(), 5);
    }

    #[test]
    fn test_boolean_buffer_conversion() {
        let data = vec![true, false, true, false, true];
        let boolean_buffer = BooleanBuffer::from(data);
        let vortex_array = boolean_buffer.into_array();
        assert_eq!(vortex_array.len(), 5);
    }

    #[test]
    fn test_scalar_buffer_conversion() {
        let data = vec![1i32, 2, 3, 4, 5];
        let scalar_buffer = ScalarBuffer::from(data);
        let vortex_array = scalar_buffer.into_array();
        assert_eq!(vortex_array.len(), 5);
    }

    #[test]
    fn test_offset_buffer_conversion() {
        let data = vec![0i32, 2, 5, 8, 10];
        let offset_buffer = OffsetBuffer::new(ScalarBuffer::from(data));
        let vortex_array = offset_buffer.into_array();
        assert_eq!(vortex_array.len(), 5);
    }

    // Test RecordBatch conversions
    #[test]
    fn test_record_batch_conversion() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("field1", DataType::Int32, false),
            Field::new("field2", DataType::Utf8, false),
        ]));

        let field1_data = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let field2_data = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));

        let record_batch = RecordBatch::try_new(schema, vec![field1_data, field2_data]).unwrap();

        let vortex_array = ArrayRef::from_arrow(record_batch, false).unwrap();
        assert_eq!(vortex_array.len(), 4);

        // Test with reference
        let schema = Arc::new(Schema::new(vec![
            Field::new("field1", DataType::Int32, false),
            Field::new("field2", DataType::Utf8, false),
        ]));

        let field1_data = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let field2_data = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));

        let record_batch = RecordBatch::try_new(schema, vec![field1_data, field2_data]).unwrap();

        let vortex_array = ArrayRef::from_arrow(&record_batch, false).unwrap();
        assert_eq!(vortex_array.len(), 4);
    }

    // Test dynamic dispatch conversion
    #[test]
    fn test_dyn_array_conversion() {
        let int_array = Int32Array::from(vec![1, 2, 3, 4]);
        let dyn_array: &dyn ArrowArray = &int_array;
        let vortex_array = ArrayRef::from_arrow(dyn_array, false).unwrap();
        assert_eq!(vortex_array.len(), 4);

        let string_array = StringArray::from(vec!["a", "b", "c"]);
        let dyn_array: &dyn ArrowArray = &string_array;
        let vortex_array = ArrayRef::from_arrow(dyn_array, false).unwrap();
        assert_eq!(vortex_array.len(), 3);

        let bool_array = BooleanArray::from(vec![true, false, true]);
        let dyn_array: &dyn ArrowArray = &bool_array;
        let vortex_array = ArrayRef::from_arrow(dyn_array, false).unwrap();
        assert_eq!(vortex_array.len(), 3);
    }

    // Existing tests
    #[test]
    pub fn nullable_may_contain_non_nullable() {
        let null_struct_array_with_non_nullable_field = new_null_array(
            &DataType::Struct(Fields::from(vec![Field::new(
                "non_nullable_inner",
                DataType::Int32,
                false,
            )])),
            1,
        );
        ArrayRef::from_arrow(null_struct_array_with_non_nullable_field.as_ref(), true).unwrap();
    }

    #[test]
    pub fn nullable_may_contain_deeply_nested_non_nullable() {
        let null_struct_array_with_non_nullable_field = new_null_array(
            &DataType::Struct(Fields::from(vec![Field::new(
                "non_nullable_inner",
                DataType::Struct(Fields::from(vec![Field::new(
                    "non_nullable_deeper_inner",
                    DataType::Int32,
                    false,
                )])),
                false,
            )])),
            1,
        );
        ArrayRef::from_arrow(null_struct_array_with_non_nullable_field.as_ref(), true).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn cannot_handle_nullable_struct_containing_non_nullable_dictionary() {
        let null_struct_array_with_non_nullable_field = new_null_array(
            &DataType::Struct(Fields::from(vec![Field::new(
                "non_nullable_deeper_inner",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            )])),
            1,
        );

        ArrayRef::from_arrow(null_struct_array_with_non_nullable_field.as_ref(), true).unwrap();
    }
}
