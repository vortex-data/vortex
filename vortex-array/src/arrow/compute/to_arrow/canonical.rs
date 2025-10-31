// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::types::{
    BinaryType, BinaryViewType, ByteArrayType, ByteViewType, Float16Type, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, LargeBinaryType, LargeUtf8Type, StringViewType,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type, Utf8Type,
};
use arrow_array::{
    Array, ArrayRef as ArrowArrayRef, ArrowPrimitiveType, BooleanArray as ArrowBoolArray,
    Decimal32Array as ArrowDecimal32Array, Decimal64Array as ArrowDecimal64Array,
    Decimal128Array as ArrowDecimal128Array, Decimal256Array as ArrowDecimal256Array,
    FixedSizeListArray as ArrowFixedSizeListArray, GenericByteArray, GenericByteViewArray,
    GenericListArray, GenericListViewArray, NullArray as ArrowNullArray, OffsetSizeTrait,
    PrimitiveArray as ArrowPrimitiveArray, StructArray as ArrowStructArray,
};
use arrow_buffer::{ScalarBuffer, i256};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field, FieldRef, Fields};
use itertools::Itertools;
use num_traits::{AsPrimitive, ToPrimitive};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, IntegerPType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::DecimalType;

use crate::arrays::{
    BoolArray, DecimalArray, FixedSizeListArray, ListViewArray, NullArray, PrimitiveArray,
    StructArray, VarBinViewArray,
};
use crate::arrow::IntoArrowArray;
use crate::arrow::array::ArrowArray;
use crate::arrow::compute::ToArrowArgs;
use crate::arrow::compute::to_arrow::null_buffer::to_null_buffer;
use crate::builders::{ArrayBuilder, ListBuilder};
use crate::compute::{InvocationArgs, Kernel, Output, cast};
use crate::{Array as _, Canonical, IntoArray, ToCanonical};

/// Implementation of `ToArrow` kernel for canonical Vortex arrays.
#[derive(Debug)]
pub(super) struct ToArrowCanonical;

impl Kernel for ToArrowCanonical {
    #[allow(clippy::cognitive_complexity)]
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ToArrowArgs {
            array,
            arrow_type: arrow_type_opt,
        } = ToArrowArgs::try_from(args)?;
        if !array.is_canonical() {
            // Not handled by this kernel
            return Ok(None);
        }

        // Figure out the target Arrow type, or use the canonical type
        let arrow_type = arrow_type_opt
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| array.dtype().to_arrow_dtype())?;

        // When `arrow_type` is `None`, conversion should respect conversion to the encoding's
        // preferred Arrow type if the array has child arrays (struct, list, and fixed-size list).
        let to_preferred = arrow_type_opt.is_none();

        let arrow_array = match (array.to_canonical(), &arrow_type) {
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
            (Canonical::Decimal(array), DataType::Decimal32(precision, scale)) => {
                if array.decimal_dtype().precision() != *precision
                    || array.decimal_dtype().scale() != *scale
                {
                    vortex_bail!(
                        "ToArrowCanonical: target precision/scale {}/{} does not match array precision/scale {}/{}",
                        precision,
                        scale,
                        array.decimal_dtype().precision(),
                        array.decimal_dtype().scale()
                    );
                }
                to_arrow_decimal32(array)
            }
            (Canonical::Decimal(array), DataType::Decimal64(precision, scale)) => {
                if array.decimal_dtype().precision() != *precision
                    || array.decimal_dtype().scale() != *scale
                {
                    vortex_bail!(
                        "ToArrowCanonical: target precision/scale {}/{} does not match array precision/scale {}/{}",
                        precision,
                        scale,
                        array.decimal_dtype().precision(),
                        array.decimal_dtype().scale()
                    );
                }
                to_arrow_decimal64(array)
            }
            (Canonical::Decimal(array), DataType::Decimal128(precision, scale)) => {
                if array.decimal_dtype().precision() != *precision
                    || array.decimal_dtype().scale() != *scale
                {
                    vortex_bail!(
                        "ToArrowCanonical: target precision/scale {}/{} does not match array precision/scale {}/{}",
                        precision,
                        scale,
                        array.decimal_dtype().precision(),
                        array.decimal_dtype().scale()
                    );
                }
                to_arrow_decimal128(array)
            }
            (Canonical::Decimal(array), DataType::Decimal256(precision, scale)) => {
                if array.decimal_dtype().precision() != *precision
                    || array.decimal_dtype().scale() != *scale
                {
                    vortex_bail!(
                        "ToArrowCanonical: target precision/scale {}/{} does not match array precision/scale {}/{}",
                        precision,
                        scale,
                        array.decimal_dtype().precision(),
                        array.decimal_dtype().scale()
                    );
                }
                to_arrow_decimal256(array)
            }
            (Canonical::Struct(array), DataType::Struct(fields)) => {
                to_arrow_struct(array, fields.as_ref(), to_preferred)
            }
            (Canonical::List(list_view), DataType::ListView(field)) => {
                to_arrow_listview::<i32>(list_view, arrow_type_opt.map(|_| field))
            }
            (Canonical::List(list_view), DataType::LargeListView(field)) => {
                to_arrow_listview::<i64>(list_view, arrow_type_opt.map(|_| field))
            }
            (Canonical::List(list_view), DataType::List(field)) => {
                to_arrow_list::<i32>(list_view, arrow_type_opt.map(|_| field))
            }
            (Canonical::List(list_view), DataType::LargeList(field)) => {
                to_arrow_list::<i64>(list_view, arrow_type_opt.map(|_| field))
            }
            (Canonical::FixedSizeList(array), DataType::FixedSizeList(field, list_size)) => {
                to_arrow_fixed_size_list(array, arrow_type_opt.map(|_| field), *list_size)
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
                array.encoding_id(),
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
        array.bit_buffer().clone().into(),
        to_null_buffer(array.validity_mask()),
    )))
}

fn to_arrow_primitive<T: ArrowPrimitiveType>(array: PrimitiveArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(array.validity_mask());
    let len = array.len();
    let buffer = array.into_byte_buffer().into_arrow_buffer();
    Ok(Arc::new(ArrowPrimitiveArray::<T>::new(
        ScalarBuffer::<T::Native>::new(buffer, 0, len),
        null_buffer,
    )))
}

fn to_arrow_decimal32(array: DecimalArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(array.validity_mask());
    let buffer: Buffer<i32> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => array.buffer::<i32>(),
        DecimalType::I64 => array
            .buffer::<i64>()
            .into_iter()
            .map(|x| {
                x.to_i32()
                    .ok_or_else(|| vortex_err!("i64 to i32 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
        DecimalType::I128 => array
            .buffer::<i128>()
            .into_iter()
            .map(|x| {
                x.to_i32()
                    .ok_or_else(|| vortex_err!("i128 to i32 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
        DecimalType::I256 => array
            .buffer::<vortex_scalar::i256>()
            .into_iter()
            .map(|x| {
                x.to_i32()
                    .ok_or_else(|| vortex_err!("i256 to i32 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
    };
    Ok(Arc::new(
        ArrowDecimal32Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_decimal64(array: DecimalArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(array.validity_mask());
    let buffer: Buffer<i64> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(array.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => array.buffer::<i64>(),
        DecimalType::I128 => array
            .buffer::<i128>()
            .into_iter()
            .map(|x| {
                x.to_i64()
                    .ok_or_else(|| vortex_err!("i128 to i64 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
        DecimalType::I256 => array
            .buffer::<vortex_scalar::i256>()
            .into_iter()
            .map(|x| {
                x.to_i64()
                    .ok_or_else(|| vortex_err!("i256 to i64 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
    };
    Ok(Arc::new(
        ArrowDecimal64Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_decimal128(array: DecimalArray) -> VortexResult<ArrowArrayRef> {
    let null_buffer = to_null_buffer(array.validity_mask());
    let buffer: Buffer<i128> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(array.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => {
            Buffer::from_trusted_len_iter(array.buffer::<i64>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I128 => array.buffer::<i128>(),
        DecimalType::I256 => array
            .buffer::<vortex_scalar::i256>()
            .into_iter()
            .map(|x| {
                x.to_i128()
                    .ok_or_else(|| vortex_err!("i256 to i128 narrowing cannot be done safely"))
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter))?,
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
    let null_buffer = to_null_buffer(array.validity_mask());
    let buffer: Buffer<i256> = match array.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(array.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(array.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(array.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => {
            Buffer::from_trusted_len_iter(array.buffer::<i64>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I128 => Buffer::from_trusted_len_iter(
            array
                .buffer::<i128>()
                .into_iter()
                .map(|x| vortex_scalar::i256::from_i128(x).into()),
        ),
        DecimalType::I256 => Buffer::<i256>::from_byte_buffer(array.byte_buffer()),
    };
    Ok(Arc::new(
        ArrowDecimal256Array::new(buffer.into_arrow_scalar_buffer(), null_buffer)
            .with_precision_and_scale(
                array.decimal_dtype().precision(),
                array.decimal_dtype().scale(),
            )?,
    ))
}

fn to_arrow_struct(
    array: StructArray,
    fields: &[FieldRef],
    to_preferred: bool,
) -> VortexResult<ArrowArrayRef> {
    if array.fields().len() != fields.len() {
        vortex_bail!(
            "StructArray has {} fields, but target Arrow type has {} fields",
            array.fields().len(),
            fields.len()
        );
    }

    let field_arrays = fields
        .iter()
        .zip_eq(array.fields().iter())
        .map(|(field, arr)| {
            // We check that the Vortex array nullability is compatible with the field
            // nullability. In other words, make sure we don't return any nulls for a
            // non-nullable field.
            if arr.dtype().is_nullable() && !field.is_nullable() && !arr.all_valid() {
                vortex_bail!(
                    "Field {} is non-nullable but has nulls {}",
                    field,
                    arr.display_tree()
                );
            }

            let result = if to_preferred {
                arr.clone().into_arrow_preferred()
            } else {
                arr.clone().into_arrow(field.data_type())
            };
            result.map_err(|err| err.with_context(format!("Failed to canonicalize field {field}")))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let nulls = to_null_buffer(array.validity_mask());

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
                name.as_ref(),
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

/// Converts a Vortex [`ListViewArray`] into an arrow [`GenericListArray`].
fn to_arrow_list<O: IntegerPType + OffsetSizeTrait>(
    array: ListViewArray,
    element_field: Option<&FieldRef>,
) -> VortexResult<ArrowArrayRef> {
    // Since `ListViewArray` can have lists stored out-of-order, we must rebuild the entire array.
    // We also can't use `list_from_list_view` because we need this specific `O` type for offsets.
    let mut list_builder = ListBuilder::<O>::with_capacity(
        array
            .dtype()
            .as_list_element_opt()
            .vortex_expect("`ListViewArray` somehow was not of type `List`")
            .clone(),
        array.dtype().nullability(),
        array.elements().len(), // This might be wrong, but it's better than nothing.
        array.len(),
    );

    // TODO(connor)[ListView]: We can potentially make a generic version of `list_from_list_view`
    // over the offsets so we don't have to rewrite this.

    list_builder.extend_from_array(&array.to_array());
    let list_array = list_builder.finish_into_list();

    // Now that we have a normal `ListArray`, we can convert all the child arrays.

    // Convert the child `elements` array to Arrow.
    let (elements, element_field) = {
        if let Some(element_field) = element_field {
            // Convert elements to the specific Arrow type the caller wants.
            let elements = list_array
                .elements()
                .clone()
                .into_arrow(element_field.data_type())?;
            let element_field = element_field.clone();
            (elements, element_field)
        } else {
            // Otherwise, convert into whatever Arrow prefers.
            let elements = list_array.elements().clone().into_arrow_preferred()?;
            let element_field = Arc::new(Field::new_list_field(
                elements.data_type().clone(),
                list_array.elements().dtype().is_nullable(),
            ));
            (elements, element_field)
        }
    };

    // Convert the child `offsets` and `validity` array to Arrow.
    let offsets = list_array
        .offsets()
        .to_primitive()
        .buffer::<O>()
        .into_arrow_offset_buffer();
    let nulls = to_null_buffer(list_array.validity_mask());

    Ok(Arc::new(GenericListArray::new(
        element_field,
        offsets,
        elements,
        nulls,
    )))
}

/// Converts a Vortex [`ListViewArray`] into an arrow [`GenericListViewArray`].
fn to_arrow_listview<O: IntegerPType + OffsetSizeTrait>(
    array: ListViewArray,
    element: Option<&FieldRef>,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the offsets and sizes into the specified width (determined by `O::PTYPE`).
    let offsets_dtype = DType::Primitive(O::PTYPE, array.dtype().nullability());
    let offsets = cast(array.offsets(), &offsets_dtype)
        .map_err(|err| err.with_context(format!("Failed to cast offsets to {offsets_dtype}")))?
        .to_primitive();
    let sizes = cast(array.sizes(), &offsets_dtype)
        .map_err(|err| err.with_context(format!("Failed to cast sizes to {offsets_dtype}")))?
        .to_primitive();

    // Convert `offsets`, `sizes`, and `validity` to Arrow buffers.
    let arrow_offsets = offsets.buffer::<O>().into_arrow_scalar_buffer();
    let arrow_sizes = sizes.buffer::<O>().into_arrow_scalar_buffer();
    let nulls = to_null_buffer(array.validity_mask());

    // Convert the child `elements` array to Arrow.
    let (elements, element_field) = {
        if let Some(element) = element {
            // Convert elements to the specific Arrow type the caller wants.
            (
                array.elements().clone().into_arrow(element.data_type())?,
                element.clone(),
            )
        } else {
            // Otherwise, convert into whatever Arrow prefers.
            let elements = array.elements().clone().into_arrow_preferred()?;
            let element_field = Arc::new(Field::new_list_field(
                elements.data_type().clone(),
                array.elements().dtype().is_nullable(),
            ));
            (elements, element_field)
        }
    };

    Ok(Arc::new(GenericListViewArray::new(
        element_field,
        arrow_offsets,
        arrow_sizes,
        elements,
        nulls,
    )))
}

fn to_arrow_fixed_size_list(
    array: FixedSizeListArray,
    element: Option<&FieldRef>,
    list_size: i32,
) -> VortexResult<ArrowArrayRef> {
    assert!(
        list_size >= 0,
        "somehow had a negative list size for arrow fixed-size lists"
    );

    if list_size as u32 != array.list_size() {
        vortex_bail!(
            "Cannot convert a Vortex `FixedSizeListArray` with list size {} to an Arrow `FixedSizeListArray` with list size {list_size}",
            array.list_size()
        );
    }

    let (values, element_field) = if let Some(element) = element {
        (
            array.elements().clone().into_arrow(element.data_type())?,
            element.clone(),
        )
    } else {
        let values = array.elements().clone().into_arrow_preferred()?;
        let element_field = Arc::new(Field::new_list_field(
            values.data_type().clone(),
            array.elements().dtype().is_nullable(),
        ));
        (values, element_field)
    };
    let nulls = to_null_buffer(array.validity_mask());

    // TODO(connor): Revert this once the issue below is resolved.
    // Ok(Arc::new(ArrowFixedSizeListArray::new(
    //     element_field,
    //     list_size,
    //     values,
    //     nulls,
    // )))

    // Build ArrayData directly to avoid the length calculation bug in try_new.
    // See: https://github.com/apache/arrow-rs/issues/8623
    let data_type = DataType::FixedSizeList(element_field, list_size);
    let list_data = ArrayData::builder(data_type)
        .len(array.len())
        .add_child_data(values.into_data())
        .nulls(nulls)
        .build()?;

    let arrow_array = ArrowFixedSizeListArray::from(list_data);

    assert_eq!(array.len(), arrow_array.len());

    Ok(Arc::new(arrow_array))
}

fn to_arrow_varbinview<T: ByteViewType>(array: VarBinViewArray) -> VortexResult<ArrowArrayRef> {
    let views =
        ScalarBuffer::<u128>::from(array.views().clone().into_byte_buffer().into_arrow_buffer());
    let buffers: Vec<_> = array
        .buffers()
        .iter()
        .map(|buffer| buffer.clone().into_arrow_buffer())
        .collect();
    let nulls = to_null_buffer(array.validity_mask());

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
    use arrow_array::{
        Array, Decimal128Array, Decimal256Array, GenericListArray, GenericListViewArray,
    };
    use arrow_buffer::i256;
    use arrow_schema::{DataType, Field};
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DecimalDType, FieldNames, NativeDecimalType};

    use crate::IntoArray;
    use crate::arrays::{DecimalArray, ListViewArray, PrimitiveArray, StructArray};
    use crate::arrow::IntoArrowArray;
    use crate::arrow::compute::to_arrow;
    use crate::builders::{ArrayBuilder, DecimalBuilder};
    use crate::validity::Validity;

    #[test]
    fn decimal_to_arrow() {
        // Make a very simple i128 and i256 array.
        let decimal_vortex = DecimalArray::new(
            buffer![1i128, 2i128, 3i128, 4i128, 5i128],
            DecimalDType::new(19, 2),
            Validity::NonNullable,
        );
        let arrow = to_arrow(decimal_vortex.as_ref(), &DataType::Decimal128(19, 2)).unwrap();
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
            FieldNames::from(["xs"]),
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
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        assert!(struct_a.into_array().into_arrow(&arrow_dt).is_err());
    }

    #[test]
    fn struct_to_arrow_with_schema_mismatch() {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::AllValid);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        let fields = vec![
            Field::new("xs", DataType::Int8, false),
            Field::new("ys", DataType::Int64, false),
        ];
        let arrow_dt = DataType::Struct(fields.into());

        let err = struct_a.into_array().into_arrow(&arrow_dt).err().unwrap();
        assert!(
            err.to_string()
                .contains("StructArray has 1 fields, but target Arrow type has 2 fields")
        );
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(vortex_scalar::i256::ZERO)]
    fn to_arrow_decimal128<T: NativeDecimalType>(#[case] _decimal_type: T) {
        let mut decimal = DecimalBuilder::new::<T>(2, 1, false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);

        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal128(2, 1)).unwrap();
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), 10);
        assert_eq!(arrow_decimal.value(1), 11);
        assert_eq!(arrow_decimal.value(2), 12);
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(vortex_scalar::i256::ZERO)]
    fn to_arrow_decimal32<T: NativeDecimalType>(#[case] _decimal_type: T) {
        use arrow_array::Decimal32Array;

        let mut decimal = DecimalBuilder::new::<T>(2, 1, false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);

        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal32(2, 1)).unwrap();
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal32Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), 10);
        assert_eq!(arrow_decimal.value(1), 11);
        assert_eq!(arrow_decimal.value(2), 12);
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(vortex_scalar::i256::ZERO)]
    fn to_arrow_decimal64<T: NativeDecimalType>(#[case] _decimal_type: T) {
        use arrow_array::Decimal64Array;

        let mut decimal = DecimalBuilder::new::<T>(2, 1, false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);

        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal64(2, 1)).unwrap();
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal64Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), 10);
        assert_eq!(arrow_decimal.value(1), 11);
        assert_eq!(arrow_decimal.value(2), 12);
    }

    #[rstest]
    #[case(0i8)]
    #[case(0i16)]
    #[case(0i32)]
    #[case(0i64)]
    #[case(0i128)]
    #[case(vortex_scalar::i256::ZERO)]
    fn to_arrow_decimal256<T: NativeDecimalType>(#[case] _decimal_type: T) {
        let mut decimal = DecimalBuilder::new::<T>(2, 1, false.into());
        decimal.append_value(10);
        decimal.append_value(11);
        decimal.append_value(12);

        let decimal = decimal.finish();

        let arrow_array = decimal.into_arrow(&DataType::Decimal256(2, 1)).unwrap();
        let arrow_decimal = arrow_array
            .as_any()
            .downcast_ref::<Decimal256Array>()
            .unwrap();
        assert_eq!(arrow_decimal.value(0), i256::from_i128(10));
        assert_eq!(arrow_decimal.value(1), i256::from_i128(11));
        assert_eq!(arrow_decimal.value(2), i256::from_i128(12));
    }

    #[test]
    fn test_to_arrow_list_i32() {
        // Create a ListViewArray with i32 elements: [[1, 2, 3], [4, 5]]
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i32, 3], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![3i32, 2], Validity::NonNullable);

        let list_array = ListViewArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        // Convert to Arrow List with i32 offsets.
        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::List(field.into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt).unwrap();

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .unwrap();

        assert_eq!(list.len(), 2);
        assert!(!list.is_null(0));
        assert!(!list.is_null(1));

        // Verify the values in the first list.
        let first_list = list.value(0);
        assert_eq!(first_list.len(), 3);
        let first_values = first_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(first_values.value(0), 1);
        assert_eq!(first_values.value(1), 2);
        assert_eq!(first_values.value(2), 3);

        // Verify the values in the second list.
        let second_list = list.value(1);
        assert_eq!(second_list.len(), 2);
        let second_values = second_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(second_values.value(0), 4);
        assert_eq!(second_values.value(1), 5);
    }

    #[test]
    fn test_to_arrow_list_i64() {
        // Create a ListViewArray with i64 offsets: [[10, 20], [30]]
        let elements = PrimitiveArray::new(buffer![10i64, 20, 30], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i64, 2], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![2i64, 1], Validity::NonNullable);

        let list_array = ListViewArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        // Convert to Arrow LargeList with i64 offsets.
        let field = Field::new("item", DataType::Int64, false);
        let arrow_dt = DataType::LargeList(field.into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt).unwrap();

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i64>>()
            .unwrap();

        assert_eq!(list.len(), 2);
        assert!(!list.is_null(0));
        assert!(!list.is_null(1));
    }

    #[test]
    fn test_to_arrow_listview_i32() {
        // Create a ListViewArray with overlapping views: [[1, 2], [2, 3], [3, 4]]
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i32, 1, 2], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![2i32, 2, 2], Validity::NonNullable);

        let list_array = ListViewArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        // Convert to Arrow ListView with i32 offsets.
        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::ListView(field.into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt).unwrap();

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let listview = arrow_array
            .as_any()
            .downcast_ref::<GenericListViewArray<i32>>()
            .unwrap();

        assert_eq!(listview.len(), 3);

        // Verify first list view [1, 2].
        let first_list = listview.value(0);
        assert_eq!(first_list.len(), 2);
        let first_values = first_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(first_values.value(0), 1);
        assert_eq!(first_values.value(1), 2);

        // Verify second list view [2, 3].
        let second_list = listview.value(1);
        assert_eq!(second_list.len(), 2);
        let second_values = second_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(second_values.value(0), 2);
        assert_eq!(second_values.value(1), 3);
    }

    #[test]
    fn test_to_arrow_listview_i64() {
        // Create a ListViewArray with nullable elements: [[100], null, [200, 300]]
        let elements = PrimitiveArray::new(buffer![100i64, 200, 300], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i64, 0, 1], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![1i64, 0, 2], Validity::NonNullable);
        let validity = Validity::from_iter([true, false, true]);

        let list_array = ListViewArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            validity,
        )
        .unwrap();

        // Convert to Arrow LargeListView with i64 offsets.
        let field = Field::new("item", DataType::Int64, false);
        let arrow_dt = DataType::LargeListView(field.into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt).unwrap();

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let listview = arrow_array
            .as_any()
            .downcast_ref::<GenericListViewArray<i64>>()
            .unwrap();

        assert_eq!(listview.len(), 3);
        assert!(!listview.is_null(0));
        assert!(listview.is_null(1));
        assert!(!listview.is_null(2));

        // Verify the third list [200, 300].
        let third_list = listview.value(2);
        assert_eq!(third_list.len(), 2);
        let third_values = third_list
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(third_values.value(0), 200);
        assert_eq!(third_values.value(1), 300);
    }
}
