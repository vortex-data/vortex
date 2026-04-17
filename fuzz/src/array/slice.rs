// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

pub fn slice_canonical_array(
    array: &ArrayRef,
    start: usize,
    stop: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let bool_buff = array.validity()?.to_mask(array.len(), ctx)?.to_bit_buffer();
        Validity::from(bool_buff.slice(start..stop))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().execute::<BoolArray>(ctx)?;
            let sliced_bools = bool_array.to_bit_buffer().slice(start..stop);
            Ok(BoolArray::new(sliced_bools, validity).into_array())
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.clone().execute::<PrimitiveArray>(ctx)?;
            match_each_native_ptype!(p, |P| {
                Ok(PrimitiveArray::new(
                    primitive_array.to_buffer::<P>().slice(start..stop),
                    validity,
                )
                .into_array())
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().execute::<VarBinViewArray>(ctx)?;
            let values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>());
            Ok(VarBinViewArray::from_iter(
                values[start..stop].iter().cloned(),
                array.dtype().clone(),
            )
            .into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.clone().execute::<StructArray>(ctx)?;
            let sliced_children = struct_array
                .iter_unmasked_fields()
                .map(|c| slice_canonical_array(c, start, stop, ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            StructArray::try_new_with_dtype(
                sliced_children,
                struct_array.struct_fields().clone(),
                stop - start,
                validity,
            )
            .map(|a| a.into_array())
        }
        DType::List(..) => {
            let list_array = array.clone().execute::<ListViewArray>(ctx)?;

            let offsets = slice_canonical_array(list_array.offsets(), start, stop, ctx)?;
            let sizes = slice_canonical_array(list_array.sizes(), start, stop, ctx)?;

            // Since the list view elements can be stored out of order, we cannot slice it.
            let elements = list_array.elements().clone();

            // SAFETY: If the array was already zero-copyable to list, slicing the offsets and sizes
            // only causes there to be leading and trailing garbage data, which is still
            // zero-copyable to a `ListArray`.
            Ok(unsafe {
                ListViewArray::new_unchecked(elements, offsets, sizes, validity)
                    .with_zero_copy_to_list(list_array.is_zero_copy_to_list())
            }
            .into_array())
        }
        DType::FixedSizeList(..) => {
            let fsl_array = array.clone().execute::<FixedSizeListArray>(ctx)?;
            let list_size = fsl_array.list_size() as usize;
            let elements = slice_canonical_array(
                fsl_array.elements(),
                start * list_size,
                stop * list_size,
                ctx,
            )?;
            let new_len = stop - start;

            FixedSizeListArray::try_new(elements, fsl_array.list_size(), validity, new_len)
                .map(|a| a.into_array())
        }
        DType::Decimal(decimal_dtype, _) => {
            let decimal_array = array.clone().execute::<DecimalArray>(ctx)?;
            Ok(
                match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                    DecimalArray::new(
                        decimal_array.buffer::<D>().slice(start..stop),
                        *decimal_dtype,
                        validity,
                    )
                })
                .into_array(),
            )
        }
        d @ (DType::Null | DType::Extension(_) | DType::Variant(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}
