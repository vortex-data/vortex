// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{
    BoolArray, DecimalArray, FixedSizeListArray, ListViewArray, PrimitiveArray, StructArray,
    VarBinViewArray,
};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_scalar::match_each_decimal_value_type;

#[allow(clippy::unnecessary_fallible_conversions)]
pub fn slice_canonical_array(
    array: &dyn Array,
    start: usize,
    stop: usize,
) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let bool_buff = array.validity_mask().to_bit_buffer();
        Validity::from(bool_buff.slice(start..stop))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool();
            let sliced_bools = bool_array.bit_buffer().slice(start..stop);
            Ok(BoolArray::from_bit_buffer(sliced_bools, validity).into_array())
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.to_primitive();
            match_each_native_ptype!(p, |P| {
                Ok(
                    PrimitiveArray::new(primitive_array.buffer::<P>().slice(start..stop), validity)
                        .into_array(),
                )
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview();
            let values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())?;
            Ok(VarBinViewArray::from_iter(
                values[start..stop].iter().cloned(),
                array.dtype().clone(),
            )
            .into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.to_struct();
            let sliced_children = struct_array
                .fields()
                .iter()
                .map(|c| slice_canonical_array(c, start, stop))
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
            let list_array = array.to_listview();

            let offsets = slice_canonical_array(list_array.offsets(), start, stop)?;
            let sizes = slice_canonical_array(list_array.sizes(), start, stop)?;

            // Since the list view elements can be stored out of order, we cannot slice it.
            let elements = list_array.elements().clone();

            ListViewArray::try_new(elements, offsets, sizes, validity).map(|a| a.into_array())
        }
        DType::FixedSizeList(..) => {
            let fsl_array = array.to_fixed_size_list();
            let list_size = fsl_array.list_size() as usize;
            let elements =
                slice_canonical_array(fsl_array.elements(), start * list_size, stop * list_size)?;
            let new_len = stop - start;

            FixedSizeListArray::try_new(elements, fsl_array.list_size(), validity, new_len)
                .map(|a| a.into_array())
        }
        DType::Decimal(decimal_dtype, _) => {
            let decimal_array = array.to_decimal();
            Ok(
                match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                    DecimalArray::new(
                        decimal_array.buffer::<D>().slice(start..stop),
                        *decimal_dtype,
                        validity,
                    )
                })
                .to_array(),
            )
        }
        d @ (DType::Null | DType::Extension(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}
