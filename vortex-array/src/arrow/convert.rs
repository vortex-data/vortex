// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::array::{
    Array as ArrowArray, ArrowPrimitiveType, BooleanArray as ArrowBooleanArray, GenericByteArray,
    NullArray as ArrowNullArray, OffsetSizeTrait, PrimitiveArray as ArrowPrimitiveArray,
    StructArray as ArrowStructArray,
};
use arrow_array::cast::{AsArray, as_null_array};
use arrow_array::types::{
    ByteArrayType, ByteViewType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
    Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{GenericByteViewArray, GenericListArray, RecordBatch, make_array};
use arrow_buffer::buffer::{NullBuffer, OffsetBuffer};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, Buffer as ArrowBuffer, ScalarBuffer};
use arrow_schema::{DataType, TimeUnit as ArrowTimeUnit};
use itertools::Itertools;
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::{DType, DecimalDType, NativePType, PType};
use vortex_error::{VortexExpect as _, vortex_panic};
use vortex_scalar::i256;

use crate::arrays::{
    BoolArray, DecimalArray, ListArray, NullArray, PrimitiveArray, StructArray, TemporalArray,
    VarBinArray, VarBinViewArray,
};
use crate::arrow::FromArrowArray;
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray};

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
        BoolArray::new(self, Validity::NonNullable).into_array()
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
    O: NativePType + OffsetSizeTrait,
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
    ($ty:path) => {
        impl FromArrowArray<&ArrowPrimitiveArray<$ty>> for ArrayRef {
            fn from_arrow(value: &ArrowPrimitiveArray<$ty>, nullable: bool) -> Self {
                let buffer = Buffer::from_arrow_scalar_buffer(value.values().clone());
                let validity = nulls(value.nulls(), nullable);
                PrimitiveArray::new(buffer, validity).into_array()
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

impl FromArrowArray<&ArrowPrimitiveArray<Decimal128Type>> for ArrayRef {
    fn from_arrow(array: &ArrowPrimitiveArray<Decimal128Type>, nullable: bool) -> Self {
        let decimal_type = DecimalDType::new(array.precision(), array.scale());
        let buffer = Buffer::from_arrow_scalar_buffer(array.values().clone());
        let validity = nulls(array.nulls(), nullable);
        DecimalArray::new(buffer, decimal_type, validity).into_array()
    }
}

impl FromArrowArray<&ArrowPrimitiveArray<Decimal256Type>> for ArrayRef {
    fn from_arrow(array: &ArrowPrimitiveArray<Decimal256Type>, nullable: bool) -> Self {
        let decimal_type = DecimalDType::new(array.precision(), array.scale());
        let buffer = Buffer::from_arrow_scalar_buffer(array.values().clone());
        // SAFETY: Our i256 implementation has the same bit-pattern representation of the
        //  arrow_buffer::i256 type. It is safe to treat values held inside the buffer as values
        //  of either type.
        let buffer =
            unsafe { std::mem::transmute::<Buffer<arrow_buffer::i256>, Buffer<i256>>(buffer) };
        let validity = nulls(array.nulls(), nullable);
        DecimalArray::new(buffer, decimal_type, validity).into_array()
    }
}

macro_rules! impl_from_arrow_temporal {
    ($ty:path) => {
        impl FromArrowArray<&ArrowPrimitiveArray<$ty>> for ArrayRef {
            fn from_arrow(value: &ArrowPrimitiveArray<$ty>, nullable: bool) -> Self {
                temporal_array(value, nullable)
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

    match T::DATA_TYPE {
        DataType::Timestamp(time_unit, tz) => {
            let tz = tz.map(|s| s.to_string());
            TemporalArray::new_timestamp(arr, time_unit.into(), tz).into()
        }
        DataType::Time32(time_unit) => TemporalArray::new_time(arr, time_unit.into()).into(),
        DataType::Time64(time_unit) => TemporalArray::new_time(arr, time_unit.into()).into(),
        DataType::Date32 => TemporalArray::new_date(arr, TimeUnit::D).into(),
        DataType::Date64 => TemporalArray::new_date(arr, TimeUnit::Ms).into(),
        DataType::Duration(_) => unimplemented!(),
        DataType::Interval(_) => unimplemented!(),
        _ => vortex_panic!("Invalid temporal type: {}", T::DATA_TYPE),
    }
}

impl<T: ByteArrayType> FromArrowArray<&GenericByteArray<T>> for ArrayRef
where
    <T as ByteArrayType>::Offset: NativePType,
{
    fn from_arrow(value: &GenericByteArray<T>, nullable: bool) -> Self {
        let dtype = match T::DATA_TYPE {
            DataType::Binary | DataType::LargeBinary => DType::Binary(nullable.into()),
            DataType::Utf8 | DataType::LargeUtf8 => DType::Utf8(nullable.into()),
            _ => vortex_panic!("Invalid data type for ByteArray: {}", T::DATA_TYPE),
        };
        VarBinArray::try_new(
            value.offsets().clone().into_array(),
            ByteBuffer::from_arrow_buffer(value.values().clone(), Alignment::of::<u8>()),
            dtype,
            nulls(value.nulls(), nullable),
        )
        .vortex_expect("Failed to convert Arrow GenericByteArray to Vortex VarBinArray")
        .into_array()
    }
}

impl<T: ByteViewType> FromArrowArray<&GenericByteViewArray<T>> for ArrayRef {
    fn from_arrow(value: &GenericByteViewArray<T>, nullable: bool) -> Self {
        let dtype = match T::DATA_TYPE {
            DataType::BinaryView => DType::Binary(nullable.into()),
            DataType::Utf8View => DType::Utf8(nullable.into()),
            _ => vortex_panic!("Invalid data type for ByteViewArray: {}", T::DATA_TYPE),
        };

        let views_buffer = Buffer::from_byte_buffer(
            Buffer::from_arrow_scalar_buffer(value.views().clone()).into_byte_buffer(),
        );

        VarBinViewArray::try_new(
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
        .vortex_expect("Failed to convert Arrow GenericByteViewArray to Vortex VarBinViewArray")
        .into_array()
    }
}

impl FromArrowArray<&ArrowBooleanArray> for ArrayRef {
    fn from_arrow(value: &ArrowBooleanArray, nullable: bool) -> Self {
        BoolArray::new(value.values().clone(), nulls(value.nulls(), nullable)).into_array()
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
    fn from_arrow(value: &ArrowStructArray, nullable: bool) -> Self {
        StructArray::try_new(
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
                .collect(),
            value.len(),
            nulls(value.nulls(), nullable),
        )
        .vortex_expect("Failed to convert Arrow StructArray to Vortex StructArray")
        .into_array()
    }
}

impl<O: OffsetSizeTrait + NativePType> FromArrowArray<&GenericListArray<O>> for ArrayRef {
    fn from_arrow(value: &GenericListArray<O>, nullable: bool) -> Self {
        // Extract the validity of the underlying element array
        let elem_nullable = match value.data_type() {
            DataType::List(field) => field.is_nullable(),
            DataType::LargeList(field) => field.is_nullable(),
            dt => vortex_panic!("Invalid data type for ListArray: {dt}"),
        };
        ListArray::try_new(
            Self::from_arrow(value.values().as_ref(), elem_nullable),
            // offsets are always non-nullable
            value.offsets().clone().into_array(),
            nulls(value.nulls(), nullable),
        )
        .vortex_expect("Failed to convert Arrow StructArray to Vortex StructArray")
        .into_array()
    }
}

impl FromArrowArray<&ArrowNullArray> for ArrayRef {
    fn from_arrow(value: &ArrowNullArray, nullable: bool) -> Self {
        assert!(nullable);
        NullArray::new(value.len()).into_array()
    }
}

fn nulls(nulls: Option<&NullBuffer>, nullable: bool) -> Validity {
    if nullable {
        nulls
            .map(|nulls| {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::from(nulls.inner().clone())
                }
            })
            .unwrap_or_else(|| Validity::AllValid)
    } else {
        assert!(nulls.map(|x| x.null_count() == 0).unwrap_or(true));
        Validity::NonNullable
    }
}

impl FromArrowArray<&dyn ArrowArray> for ArrayRef {
    fn from_arrow(array: &dyn ArrowArray, nullable: bool) -> Self {
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
                _ => unreachable!(),
            },
            DataType::Time64(u) => match u {
                ArrowTimeUnit::Microsecond => {
                    Self::from_arrow(array.as_primitive::<Time64MicrosecondType>(), nullable)
                }
                ArrowTimeUnit::Nanosecond => {
                    Self::from_arrow(array.as_primitive::<Time64NanosecondType>(), nullable)
                }
                _ => unreachable!(),
            },
            DataType::Decimal128(..) => {
                Self::from_arrow(array.as_primitive::<Decimal128Type>(), nullable)
            }
            DataType::Decimal256(..) => {
                Self::from_arrow(array.as_primitive::<Decimal256Type>(), nullable)
            }
            _ => vortex_panic!(
                "Array encoding not implemented for Arrow data type {}",
                array.data_type().clone()
            ),
        }
    }
}

impl FromArrowArray<RecordBatch> for ArrayRef {
    fn from_arrow(array: RecordBatch, nullable: bool) -> Self {
        ArrayRef::from_arrow(&arrow_array::StructArray::from(array), nullable)
    }
}

impl FromArrowArray<&RecordBatch> for ArrayRef {
    fn from_arrow(array: &RecordBatch, nullable: bool) -> Self {
        Self::from_arrow(array.clone(), nullable)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{
        BinaryViewBuilder, Decimal128Builder, Decimal256Builder, Int32Builder, LargeListBuilder,
        ListBuilder, StringViewBuilder,
    };
    use arrow_array::types::{ArrowPrimitiveType, Float16Type};
    use arrow_array::{
        Array as ArrowArray, BinaryArray, BooleanArray, Date32Array, Date64Array, Float32Array,
        Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
        LargeStringArray, NullArray, RecordBatch, StringArray, StructArray, Time32MillisecondArray,
        Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_null_array,
    };
    use arrow_buffer::{BooleanBuffer, Buffer as ArrowBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use vortex_dtype::datetime::TimeUnit;
    use vortex_dtype::{DType, PType};

    use crate::arrays::{
        DecimalVTable, ListVTable, PrimitiveVTable, StructVTable, TemporalArray, VarBinVTable,
        VarBinViewVTable,
    };
    use crate::arrow::FromArrowArray as _;
    use crate::{ArrayRef, IntoArray};

    // Test primitive array conversions
    #[test]
    fn test_int8_array_conversion() {
        let arrow_array = Int8Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Int8Array::from(vec![1, 2, 3, 4]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I8 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::I8);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I8);
    }

    #[test]
    fn test_int16_array_conversion() {
        let arrow_array = Int16Array::from(vec![Some(100), None, Some(300), Some(400)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Int16Array::from(vec![100, 200, 300, 400]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I16 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::I16);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I16);
    }

    #[test]
    fn test_int32_array_conversion() {
        let arrow_array = Int32Array::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Int32Array::from(vec![1000, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I32 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::I32);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I32);
    }

    #[test]
    fn test_int64_array_conversion() {
        let arrow_array = Int64Array::from(vec![Some(10000), None, Some(30000), Some(40000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Int64Array::from(vec![10000_i64, 20000, 30000, 40000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with I64 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::I64);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::I64);
    }

    #[test]
    fn test_uint8_array_conversion() {
        let arrow_array = UInt8Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = UInt8Array::from(vec![1_u8, 2, 3, 4]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U8 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::U8);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U8);
    }

    #[test]
    fn test_uint16_array_conversion() {
        let arrow_array = UInt16Array::from(vec![Some(100), None, Some(300), Some(400)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = UInt16Array::from(vec![100_u16, 200, 300, 400]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U16 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::U16);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U16);
    }

    #[test]
    fn test_uint32_array_conversion() {
        let arrow_array = UInt32Array::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = UInt32Array::from(vec![1000_u32, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U32 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::U32);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::U32);
    }

    #[test]
    fn test_uint64_array_conversion() {
        let arrow_array = UInt64Array::from(vec![Some(10000), None, Some(30000), Some(40000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = UInt64Array::from(vec![10000_u64, 20000, 30000, 40000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with U64 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::U64);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let non_null_values = vec![
            <Float16Type as ArrowPrimitiveType>::Native::from_f32(1.5),
            <Float16Type as ArrowPrimitiveType>::Native::from_f32(2.5),
        ];
        let arrow_array_non_null =
            arrow_array::PrimitiveArray::<Float16Type>::from(non_null_values);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 3);
        assert_eq!(vortex_array_non_null.len(), 2);

        // Verify metadata - should be PrimitiveArray with F16 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::F16);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::F16);
    }

    #[test]
    fn test_float32_array_conversion() {
        let arrow_array = Float32Array::from(vec![Some(1.5), None, Some(3.5), Some(4.5)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Float32Array::from(vec![1.5_f32, 2.5, 3.5, 4.5]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with F32 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::F32);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array_non_null.ptype(), PType::F32);
    }

    #[test]
    fn test_float64_array_conversion() {
        let arrow_array = Float64Array::from(vec![Some(1.5), None, Some(3.5), Some(4.5)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Float64Array::from(vec![1.5_f64, 2.5, 3.5, 4.5]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be PrimitiveArray with F64 ptype
        let primitive_array = vortex_array.as_::<PrimitiveVTable>();
        assert_eq!(primitive_array.ptype(), PType::F64);

        let primitive_array_non_null = vortex_array_non_null.as_::<PrimitiveVTable>();
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

        let vortex_array = ArrayRef::from_arrow(&decimal_array, true);
        assert_eq!(vortex_array.len(), 4);

        let mut builder_non_null = Decimal128Builder::with_capacity(3);
        builder_non_null.append_value(12345);
        builder_non_null.append_value(67890);
        builder_non_null.append_value(11111);
        let decimal_array_non_null = builder_non_null
            .finish()
            .with_precision_and_scale(10, 2)
            .unwrap();

        let vortex_array_non_null = ArrayRef::from_arrow(&decimal_array_non_null, false);
        assert_eq!(vortex_array_non_null.len(), 3);

        // Verify metadata - should be DecimalArray with correct precision and scale
        let decimal_vortex_array = vortex_array.as_::<DecimalVTable>();
        assert_eq!(decimal_vortex_array.decimal_dtype().precision(), 10);
        assert_eq!(decimal_vortex_array.decimal_dtype().scale(), 2);

        let decimal_vortex_array_non_null = vortex_array_non_null.as_::<DecimalVTable>();
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

        let vortex_array = ArrayRef::from_arrow(&decimal_array, true);
        assert_eq!(vortex_array.len(), 4);

        let mut builder_non_null = Decimal256Builder::with_capacity(3);
        builder_non_null.append_value(arrow_buffer::i256::from_i128(12345));
        builder_non_null.append_value(arrow_buffer::i256::from_i128(67890));
        builder_non_null.append_value(arrow_buffer::i256::from_i128(11111));
        let decimal_array_non_null = builder_non_null
            .finish()
            .with_precision_and_scale(38, 10)
            .unwrap();

        let vortex_array_non_null = ArrayRef::from_arrow(&decimal_array_non_null, false);
        assert_eq!(vortex_array_non_null.len(), 3);

        // Verify metadata - should be DecimalArray with correct precision and scale
        let decimal_vortex_array = vortex_array.as_::<DecimalVTable>();
        assert_eq!(decimal_vortex_array.decimal_dtype().precision(), 38);
        assert_eq!(decimal_vortex_array.decimal_dtype().scale(), 10);

        let decimal_vortex_array_non_null = vortex_array_non_null.as_::<DecimalVTable>();
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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = TimestampSecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be TemporalArray with Second time unit
        let temporal_array = TemporalArray::try_from(vortex_array.clone()).unwrap();
        assert_eq!(temporal_array.temporal_metadata().time_unit(), TimeUnit::S);

        let temporal_array_non_null =
            TemporalArray::try_from(vortex_array_non_null.clone()).unwrap();
        assert_eq!(
            temporal_array_non_null.temporal_metadata().time_unit(),
            TimeUnit::S
        );
    }

    #[test]
    fn test_timestamp_millisecond_array_conversion() {
        let arrow_array =
            TimestampMillisecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null =
            TimestampMillisecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_timestamp_microsecond_array_conversion() {
        let arrow_array =
            TimestampMicrosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null =
            TimestampMicrosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_timestamp_nanosecond_array_conversion() {
        let arrow_array =
            TimestampNanosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = TimestampNanosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_time32_second_array_conversion() {
        let arrow_array = Time32SecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Time32SecondArray::from(vec![1000_i32, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be TemporalArray with Second time unit
        let temporal_array = TemporalArray::try_from(vortex_array.clone()).unwrap();
        assert_eq!(temporal_array.temporal_metadata().time_unit(), TimeUnit::S);

        let temporal_array_non_null =
            TemporalArray::try_from(vortex_array_non_null.clone()).unwrap();
        assert_eq!(
            temporal_array_non_null.temporal_metadata().time_unit(),
            TimeUnit::S
        );
    }

    #[test]
    fn test_time32_millisecond_array_conversion() {
        let arrow_array =
            Time32MillisecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Time32MillisecondArray::from(vec![1000_i32, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_time64_microsecond_array_conversion() {
        let arrow_array =
            Time64MicrosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Time64MicrosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_time64_nanosecond_array_conversion() {
        let arrow_array =
            Time64NanosecondArray::from(vec![Some(1000), None, Some(3000), Some(4000)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Time64NanosecondArray::from(vec![1000_i64, 2000, 3000, 4000]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    #[test]
    fn test_date32_array_conversion() {
        let arrow_array = Date32Array::from(vec![Some(18000), None, Some(18002), Some(18003)]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Date32Array::from(vec![18000_i32, 18001, 18002, 18003]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = Date64Array::from(vec![
            1555200000000_i64,
            1555213600000,
            1555286400000,
            1555372800000,
        ]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);
    }

    // Test string/binary array conversions
    #[test]
    fn test_utf8_array_conversion() {
        let arrow_array = StringArray::from(vec![Some("hello"), None, Some("world"), Some("test")]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = StringArray::from(vec!["hello", "world", "test", "vortex"]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be VarBinArray with Utf8 dtype
        let varbin_array = vortex_array.as_::<VarBinVTable>();
        assert_eq!(varbin_array.dtype(), &DType::Utf8(true.into()));

        let varbin_array_non_null = vortex_array_non_null.as_::<VarBinVTable>();
        assert_eq!(varbin_array_non_null.dtype(), &DType::Utf8(false.into()));
    }

    #[test]
    fn test_large_utf8_array_conversion() {
        let arrow_array =
            LargeStringArray::from(vec![Some("hello"), None, Some("world"), Some("test")]);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = LargeStringArray::from(vec!["hello", "world", "test", "vortex"]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = BinaryArray::from(vec![
            "hello".as_bytes(),
            "world".as_bytes(),
            "test".as_bytes(),
            "vortex".as_bytes(),
        ]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = LargeBinaryArray::from(vec![
            "hello".as_bytes(),
            "world".as_bytes(),
            "test".as_bytes(),
            "vortex".as_bytes(),
        ]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let mut builder_non_null = StringViewBuilder::new();
        builder_non_null.append_value("hello");
        builder_non_null.append_value("world");
        builder_non_null.append_value("test");
        builder_non_null.append_value("vortex");
        let arrow_array_non_null = builder_non_null.finish();
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be VarBinViewArray with correct buffer count and dtype
        let varbin_view_array = vortex_array.as_::<VarBinViewVTable>();
        assert_eq!(
            varbin_view_array.buffers().len(),
            arrow_array.data_buffers().len()
        );
        assert_eq!(varbin_view_array.dtype(), &DType::Utf8(true.into()));

        let varbin_view_array_non_null = vortex_array_non_null.as_::<VarBinViewVTable>();
        assert_eq!(
            varbin_view_array_non_null.buffers().len(),
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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let mut builder_non_null = BinaryViewBuilder::new();
        builder_non_null.append_value(b"hello");
        builder_non_null.append_value(b"world");
        builder_non_null.append_value(b"test");
        builder_non_null.append_value(b"vortex");
        let arrow_array_non_null = builder_non_null.finish();
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

        assert_eq!(vortex_array.len(), 4);
        assert_eq!(vortex_array_non_null.len(), 4);

        // Verify metadata - should be VarBinViewArray with correct buffer count and dtype
        let varbin_view_array = vortex_array.as_::<VarBinViewVTable>();
        assert_eq!(
            varbin_view_array.buffers().len(),
            arrow_array.data_buffers().len()
        );
        assert_eq!(varbin_view_array.dtype(), &DType::Binary(true.into()));

        let varbin_view_array_non_null = vortex_array_non_null.as_::<VarBinViewVTable>();
        assert_eq!(
            varbin_view_array_non_null.buffers().len(),
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
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);

        let arrow_array_non_null = BooleanArray::from(vec![true, false, true, false]);
        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);

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

        let vortex_array = ArrayRef::from_arrow(&arrow_array, false);
        assert_eq!(vortex_array.len(), 3);

        // Verify metadata - should be StructArray with correct field names
        let struct_vortex_array = vortex_array.as_::<StructVTable>();
        assert_eq!(struct_vortex_array.names().len(), 2);
        assert_eq!(struct_vortex_array.names()[0], "field1".into());
        assert_eq!(struct_vortex_array.names()[1], "field2".into());

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

        let vortex_nullable_array = ArrayRef::from_arrow(&nullable_array, true);
        assert_eq!(vortex_nullable_array.len(), 3);

        // Verify metadata for nullable struct
        let struct_vortex_nullable_array = vortex_nullable_array.as_::<StructVTable>();
        assert_eq!(struct_vortex_nullable_array.names().len(), 2);
        assert_eq!(struct_vortex_nullable_array.names()[0], "field1".into());
        assert_eq!(struct_vortex_nullable_array.names()[1], "field2".into());
    }

    // Test list array conversions
    #[test]
    fn test_list_array_conversion() {
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.append_value([Some(1), None, Some(3)]);
        builder.append_null();
        builder.append_value([Some(4), Some(5)]);
        let arrow_array = builder.finish();

        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);
        assert_eq!(vortex_array.len(), 3);

        // Verify metadata - should be ListArray with correct offsets
        let list_vortex_array = vortex_array.as_::<ListVTable>();
        let offsets_array = list_vortex_array.offsets().as_::<PrimitiveVTable>();
        assert_eq!(offsets_array.len(), 4); // n+1 offsets for n lists
        assert_eq!(offsets_array.ptype(), PType::I32);

        // Test non-nullable list
        let mut builder_non_null = ListBuilder::new(Int32Builder::new());
        builder_non_null.append_value([Some(1), None, Some(3)]);
        builder_non_null.append_value([Some(4), Some(5)]);
        let arrow_array_non_null = builder_non_null.finish();

        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);
        assert_eq!(vortex_array_non_null.len(), 2);

        // Verify metadata for non-nullable list
        let list_vortex_array_non_null = vortex_array_non_null.as_::<ListVTable>();
        let offsets_array_non_null = list_vortex_array_non_null
            .offsets()
            .as_::<PrimitiveVTable>();
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

        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);
        assert_eq!(vortex_array.len(), 3);

        // Verify metadata - should be ListArray with correct offsets (I64 for large lists)
        let list_vortex_array = vortex_array.as_::<ListVTable>();
        let offsets_array = list_vortex_array.offsets().as_::<PrimitiveVTable>();
        assert_eq!(offsets_array.len(), 4); // n+1 offsets for n lists
        assert_eq!(offsets_array.ptype(), PType::I64); // Large lists use I64 offsets

        // Test non-nullable large list
        let mut builder_non_null = LargeListBuilder::new(Int32Builder::new());
        builder_non_null.append_value([Some(1), None, Some(3)]);
        builder_non_null.append_value([Some(4), Some(5)]);
        let arrow_array_non_null = builder_non_null.finish();

        let vortex_array_non_null = ArrayRef::from_arrow(&arrow_array_non_null, false);
        assert_eq!(vortex_array_non_null.len(), 2);

        // Verify metadata for non-nullable large list
        let list_vortex_array_non_null = vortex_array_non_null.as_::<ListVTable>();
        let offsets_array_non_null = list_vortex_array_non_null
            .offsets()
            .as_::<PrimitiveVTable>();
        assert_eq!(offsets_array_non_null.len(), 3); // n+1 offsets for n lists
        assert_eq!(offsets_array_non_null.ptype(), PType::I64); // Large lists use I64 offsets
    }

    // Test null array conversions
    #[test]
    fn test_null_array_conversion() {
        let arrow_array = NullArray::new(5);
        let vortex_array = ArrayRef::from_arrow(&arrow_array, true);
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

        let vortex_array = ArrayRef::from_arrow(record_batch, false);
        assert_eq!(vortex_array.len(), 4);

        // Test with reference
        let schema = Arc::new(Schema::new(vec![
            Field::new("field1", DataType::Int32, false),
            Field::new("field2", DataType::Utf8, false),
        ]));

        let field1_data = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let field2_data = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));

        let record_batch = RecordBatch::try_new(schema, vec![field1_data, field2_data]).unwrap();

        let vortex_array = ArrayRef::from_arrow(&record_batch, false);
        assert_eq!(vortex_array.len(), 4);
    }

    // Test dynamic dispatch conversion
    #[test]
    fn test_dyn_array_conversion() {
        let int_array = Int32Array::from(vec![1, 2, 3, 4]);
        let dyn_array: &dyn ArrowArray = &int_array;
        let vortex_array = ArrayRef::from_arrow(dyn_array, false);
        assert_eq!(vortex_array.len(), 4);

        let string_array = StringArray::from(vec!["a", "b", "c"]);
        let dyn_array: &dyn ArrowArray = &string_array;
        let vortex_array = ArrayRef::from_arrow(dyn_array, false);
        assert_eq!(vortex_array.len(), 3);

        let bool_array = BooleanArray::from(vec![true, false, true]);
        let dyn_array: &dyn ArrowArray = &bool_array;
        let vortex_array = ArrayRef::from_arrow(dyn_array, false);
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
        ArrayRef::from_arrow(null_struct_array_with_non_nullable_field.as_ref(), true);
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
        ArrayRef::from_arrow(null_struct_array_with_non_nullable_field.as_ref(), true);
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

        ArrayRef::from_arrow(null_struct_array_with_non_nullable_field.as_ref(), true);
    }
}
