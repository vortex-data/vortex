// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::sync::Arc;

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

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
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>());
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
        DType::Extension(ext_dtype) => {
            // Extension arrays delegate slicing to their storage type
            let sliced_storage =
                slice_canonical_array(array.to_extension().storage(), start, stop)?;

            if sliced_storage.dtype().nullability() == ext_dtype.storage_dtype().nullability() {
                Ok(ExtensionArray::new(ext_dtype.clone(), sliced_storage).into_array())
            } else {
                let new_ext_dtype = Arc::new(ExtDType::new(
                    ext_dtype.id().clone(),
                    Arc::new(sliced_storage.dtype().clone()),
                    ext_dtype.metadata().cloned(),
                ));
                Ok(ExtensionArray::new(new_ext_dtype, sliced_storage).into_array())
            }
        }
        DType::Null => {
            unreachable!("Cannot search sorted on Null array")
        }
    }
}
