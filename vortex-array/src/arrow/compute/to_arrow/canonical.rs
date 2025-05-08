use std::sync::Arc;

use arrow_array::types::{
    BinaryType, BinaryViewType, ByteArrayType, ByteViewType, Float16Type, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, LargeBinaryType, LargeUtf8Type, StringViewType,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type, Utf8Type,
};
use arrow_array::{
    Array, ArrayRef as ArrowArrayRef, ArrowPrimitiveType, BooleanArray as ArrowBoolArray,
    Decimal128Array as ArrowDecimal128Array, Decimal256Array as ArrowDecimal256Array,
    GenericByteArray, GenericByteViewArray, GenericListArray, NullArray as ArrowNullArray,
    OffsetSizeTrait, PrimitiveArray as ArrowPrimitiveArray, StructArray as ArrowStructArray,
};
use arrow_buffer::{ScalarBuffer, i256};
use arrow_schema::{DataType, Field, FieldRef, Fields};
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{
    BoolArray, DecimalArray, DecimalValueType, ListArray, NullArray, PrimitiveArray, StructArray,
    VarBinViewArray,
};
use crate::arrow::IntoArrowArray;
use crate::arrow::array::ArrowArray;
use crate::arrow::compute::ToArrowArgs;
use crate::compute::{InvocationArgs, Kernel, Output, cast};
use crate::variants::{PrimitiveArrayTrait, StructArrayTrait};
use crate::{Array as _, Canonical, ToCanonical};

/// Implementation of `ToArrow` kernel for canonical Vortex arrays.
#[derive(Debug)]
pub(super) struct ToArrowCanonical;

impl Kernel for ToArrowCanonical {
    #[allow(clippy::cognitive_complexity)]
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ToArrowArgs { array, arrow_type } = ToArrowArgs::try_from(args)?;
        if !array.is_canonical() {
            // Not handled by this kernel
            return Ok(None);
        }

        // Figure out the target Arrow type, or use the canonical type
        let arrow_type = arrow_type
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| array.dtype().to_arrow_dtype())?;

        let arrow_array = match (array.to_canonical()?, &arrow_type) {
            (Canonical::Null(array), DataType::Null) => to_arrow_null(array),
            (Canonical::Bool(array), DataType::Boolean) => to_arrow_bool(array),
            (Canonical::Primitive(array), DataType::Int8) if matches!(array.ptype(), PType::I8) => {
                to_arrow_primitive::<Int8Type>(array)
            }
            (Canonical::Primitive(array), DataType::Int16)
                if matches!(array.ptype(), PType::I16) =>
            {
                to_arrow_primitive::<Int16Type>(array)
            }
            (Canonical::Primitive(array), DataType::Int32)
                if matches!(array.ptype(), PType::I32) =>
            {
                to_arrow_primitive::<Int32Type>(array)
            }
            (Canonical::Primitive(array), DataType::Int64)
                if matches!(array.ptype(), PType::I64) =>
            {
                to_arrow_primitive::<Int64Type>(array)
            }
            (Canonical::Primitive(array), DataType::UInt8)
                if matches!(array.ptype(), PType::U8) =>
            {
                to_arrow_primitive::<UInt8Type>(array)
            }
            (Canonical::Primitive(array), DataType::UInt16)
                if matches!(array.ptype(), PType::U16) =>
            {
                to_arrow_primitive::<UInt16Type>(array)
            }
            (Canonical::Primitive(array), DataType::UInt32)
                if matches!(array.ptype(), PType::U32) =>
            {
                to_arrow_primitive::<UInt32Type>(array)
            }
            (Canonical::Primitive(array), DataType::UInt64)
                if matches!(array.ptype(), PType::U64) =>
            {
                to_arrow_primitive::<UInt64Type>(array)
            }
            (Canonical::Primitive(array), DataType::Float16)
                if matches!(array.ptype(), PType::F16) =>
            {
                to_arrow_primitive::<Float16Type>(array)
            }
            (Canonical::Primitive(array), DataType::Float32)
                if matches!(array.ptype(), PType::F32) =>
            {
                to_arrow_primitive::<Float32Type>(array)
            }
            (Canonical::Primitive(array), DataType::Float64)
                if matches!(array.ptype(), PType::F64) =>
            {
                to_arrow_primitive::<Float64Type>(array)
            }
            (Canonical::Decimal(array), DataType::Decimal128(..)) => to_arrow_decimal128(array),
            (Canonical::Decimal(array), DataType::Decimal256(..)) => to_arrow_decimal256(array),
            (Canonical::Struct(array), DataType::Struct(fields)) => {
                to_arrow_struct(array, fields.as_ref())
            }
            (Canonical::List(array), DataType::List(field)) => to_arrow_list::<i32>(array, field),
            (Canonical::List(array), DataType::LargeList(field)) => {
                to_arrow_list::<i64>(array, field)
            }
            (Canonical::VarBinView(array), DataType::BinaryView) if array.dtype().is_binary() => {
                to_arrow_varbinview::<BinaryViewType>(array)
            }
            (Canonical::VarBinView(array), DataType::Binary) if array.dtype().is_binary() => {
                to_arrow_varbin::<BinaryViewType, BinaryType>(
                    to_arrow_varbinview::<BinaryViewType>(array)?,
                )
            }
            (Canonical::VarBinView(array), DataType::LargeBinary) if array.dtype().is_binary() => {
                to_arrow_varbin::<BinaryViewType, LargeBinaryType>(to_arrow_varbinview::<
                    BinaryViewType,
                >(array)?)
            }
            (Canonical::VarBinView(array), DataType::Utf8View) if array.dtype().is_utf8() => {
                to_arrow_varbinview::<StringViewType>(array)
            }
            (Canonical::VarBinView(array), DataType::Utf8) if array.dtype().is_utf8() => {
                to_arrow_varbin::<StringViewType, Utf8Type>(to_arrow_varbinview::<StringViewType>(
                    array,
                )?)
            }
            (Canonical::VarBinView(array), DataType::LargeUtf8) if array.dtype().is_utf8() => {
                to_arrow_varbin::<StringViewType, LargeUtf8Type>(to_arrow_varbinview::<
                    StringViewType,
                >(array)?)
            }
            (Canonical::Extension(_), _) => {
                // Datetime and interval types are handled by a different kernel.
                return Ok(None);
            }
            _ => vortex_bail!(
                "Cannot convert canonical array {} with dtype {} to: {:?}",
                array.encoding(),
                array.dtype(),
                &arrow_type
            ),
        }?;

        Ok(Some(
            ArrowArray::new(arrow_array, array.dtype().nullability())
                .into_array()
                .into(),
        ))
    }
}

fn to_arrow_null(array: NullArray) -> VortexResult<ArrowArrayRef> {
    Ok(Arc::new(ArrowNullArray::new(array.len())))
}

fn to_arrow_bool(array: BoolArray) -> VortexResult<ArrowArrayRef> {
    Ok(Arc::new(ArrowBoolArray::new(
        array.boolean_buffer().clone(),
        array.validity_mask()?.to_null_buffer(),
    )))
}

fn to_arrow_primitive<T: ArrowPrimitiveType>(array: PrimitiveArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = array.validity_mask()?.to_null_buffer();
    let len = array.len();
    let buffer = array.into_byte_buffer().into_arrow_buffer();
    Ok(Arc::new(ArrowPrimitiveArray::<T>::new(
        ScalarBuffer::<T::Native>::new(buffer, 0, len),
        null_buffer,
    )))
}

fn to_arrow_decimal128(array: DecimalArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = array.validity_mask()?.to_null_buffer();
    let buffer: Buffer<i128> = match array.values_type() {
        DecimalValueType::I8 => array.buffer::<i8>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I16 => array.buffer::<i16>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I32 => array.buffer::<i32>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I64 => array.buffer::<i64>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I128 => array.buffer::<i128>(),
        DecimalValueType::I256 => {
            vortex_bail!("i256 decimals cannot be converted to Arrow i128 decimal")
        }
    };
    Ok(Arc::new(
        ArrowDecimal128Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_decimal256(array: DecimalArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = array.validity_mask()?.to_null_buffer();
    let buffer: Buffer<i256> = match array.values_type() {
        DecimalValueType::I8 => array.buffer::<i8>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I16 => array.buffer::<i8>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I32 => array.buffer::<i8>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I64 => array.buffer::<i8>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I128 => array.buffer::<i8>().into_iter().map(|x| x.as_()).collect(),
        DecimalValueType::I256 => Buffer::<i256>::from_byte_buffer(array.byte_buffer()),
    };
    Ok(Arc::new(
        ArrowDecimal256Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_struct(array: StructArray, fields: &[FieldRef]) -> VortexResult<ArrowArrayRef> {
    let field_arrays = fields
        .iter()
        .zip_eq(array.fields())
        .map(|(field, arr)| {
            // We check that the Vortex array nullability is compatible with the field
            // nullability. In other words, make sure we don't return any nulls for a
            // non-nullable field.
            if arr.dtype().is_nullable() && !field.is_nullable() && !arr.all_valid()? {
                vortex_bail!(
                    "Field {} is non-nullable but has nulls {}",
                    field,
                    arr.tree_display()
                );
            }

            arr.clone()
                .into_arrow(field.data_type())
                .map_err(|err| err.with_context(format!("Failed to canonicalize field {}", field)))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let nulls = array.validity_mask()?.to_null_buffer();

    if field_arrays.is_empty() {
        return Ok(Arc::new(ArrowStructArray::new_empty_fields(
            array.len(),
            nulls,
        )));
    }

    let arrow_fields = array
        .names()
        .iter()
        .zip(field_arrays.iter())
        .zip(fields.iter())
        .map(|((name, field_array), target_field)| {
            Field::new(
                &**name,
                field_array.data_type().clone(),
                target_field.is_nullable(),
            )
        })
        .map(Arc::new)
        .collect::<Fields>();

    Ok(Arc::new(ArrowStructArray::try_new(
        arrow_fields,
        field_arrays,
        nulls,
    )?))
}

fn to_arrow_list<O: NativePType + OffsetSizeTrait>(
    array: ListArray,
    element: &FieldRef,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the offsets into the correct width.
    let offsets_dtype = DType::Primitive(O::PTYPE, array.dtype().nullability());
    let arrow_offsets = cast(array.offsets(), &offsets_dtype)
        .map_err(|err| err.with_context(format!("Failed to cast offsets to {}", offsets_dtype)))?
        .to_primitive()?;

    let values = array.elements().clone().into_arrow(element.data_type())?;
    let nulls = array.validity_mask()?.to_null_buffer();

    Ok(Arc::new(GenericListArray::new(
        element.clone(),
        arrow_offsets.buffer::<O>().into_arrow_offset_buffer(),
        values,
        nulls,
    )))
}

fn to_arrow_varbinview<T: ByteViewType>(array: VarBinViewArray) -> VortexResult<ArrowArrayRef> {
    let views =
        ScalarBuffer::<u128>::from(array.views().clone().into_byte_buffer().into_arrow_buffer());
    let buffers: Vec<_> = array
        .buffers()
        .iter()
        .map(|buffer| buffer.clone().into_arrow_buffer())
        .collect();
    let nulls = array
        .validity_mask()
        .vortex_expect("VarBinViewArray: failed to get logical validity")
        .to_null_buffer();

    // SAFETY: our own VarBinView array is considered safe.
    Ok(Arc::new(unsafe {
        GenericByteViewArray::<T>::new_unchecked(views, buffers, nulls)
    }))
}

fn to_arrow_varbin<V: ByteViewType, T: ByteArrayType>(
    arrow_varbinview: ArrowArrayRef,
) -> VortexResult<ArrowArrayRef>
where
    <V as ByteViewType>::Native: AsRef<<T as ByteArrayType>::Native>,
{
    let varbinview = arrow_varbinview
        .as_any()
        .downcast_ref::<GenericByteViewArray<V>>()
        .vortex_expect("VarBinViewArray: failed to downcast to GenericByteViewArray");

    // Note that this conversion requires a copy.
    Ok(Arc::new(GenericByteArray::<T>::from_iter(
        varbinview.iter(),
    )))
}

#[cfg(test)]
mod tests {
    use arrow_array::Decimal128Array;
    use arrow_schema::{DataType, Field};
    use vortex_buffer::buffer;
    use vortex_dtype::{DecimalDType, FieldNames};

    use crate::Array as _;
    use crate::arrays::{DecimalArray, PrimitiveArray, StructArray};
    use crate::arrow::IntoArrowArray;
    use crate::arrow::compute::to_arrow;
    use crate::validity::Validity;

    #[test]
    fn decimal_to_arrow() {
        // Make a very simple i128 and i256 array.
        let decimal_vortex = DecimalArray::new(
            buffer![1i128, 2i128, 3i128, 4i128, 5i128],
            DecimalDType::new(19, 2),
            Validity::NonNullable,
        );
        let arrow = to_arrow(&decimal_vortex, &DataType::Decimal128(19, 2)).unwrap();
        assert_eq!(arrow.data_type(), &DataType::Decimal128(19, 2));
        let decimal_array = arrow.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(
            decimal_array.values().as_ref(),
            &[1i128, 2i128, 3i128, 4i128, 5i128]
        );
    }

    #[test]
    fn struct_nullable_non_null_to_arrow() {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::AllValid);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into()]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        struct_a.into_array().into_arrow(&arrow_dt).unwrap();
    }

    #[test]
    fn struct_nullable_with_nulls_to_arrow() {
        let xs =
            PrimitiveArray::from_option_iter(vec![Some(0_i64), Some(1), Some(2), None, Some(3)]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into()]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        assert!(struct_a.into_array().into_arrow(&arrow_dt).is_err());
    }
}
