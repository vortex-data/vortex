// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::array::take_canonical_array_non_nullable_indices;

pub fn filter_canonical_array(array: &ArrayRef, filter: &[bool]) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let validity_buff = array.validity_mask()?.to_bit_buffer();
        Validity::from_iter(
            filter
                .iter()
                .zip(validity_buff.iter())
                .filter(|(f, _)| **f)
                .map(|(_, v)| v),
        )
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool();
            Ok(BoolArray::new(
                BitBuffer::from_iter(
                    filter
                        .iter()
                        .zip(bool_array.to_bit_buffer().iter())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v),
                ),
                validity,
            )
            .into_array())
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |P| {
            let primitive_array = array.to_primitive();
            Ok(PrimitiveArray::new(
                filter
                    .iter()
                    .zip(primitive_array.as_slice::<P>().iter().copied())
                    .filter(|(f, _)| **f)
                    .map(|(_, v)| v)
                    .collect::<Buffer<_>>(),
                validity,
            )
            .into_array())
        }),
        DType::Decimal(d, _) => {
            let decimal_array = array.to_decimal();
            match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                let buf = decimal_array.buffer::<D>();
                Ok(DecimalArray::new(
                    filter
                        .iter()
                        .zip(buf.as_slice().iter().copied())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v)
                        .collect::<Buffer<_>>(),
                    *d,
                    validity,
                )
                .into_array())
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview();
            let values = utf8.with_iterator(|iter| {
                iter.zip(filter.iter())
                    .filter(|(_, f)| **f)
                    .map(|(v, _)| v.map(|u| u.to_vec()))
                    .collect::<Vec<_>>()
            });
            Ok(VarBinViewArray::from_iter(values, array.dtype().clone()).into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.to_struct();
            let filtered_children = struct_array
                .iter_unmasked_fields()
                .map(|c| filter_canonical_array(c, filter))
                .collect::<VortexResult<Vec<_>>>()?;

            StructArray::try_new_with_dtype(
                filtered_children,
                struct_array.struct_fields().clone(),
                filter.iter().filter(|b| **b).map(|b| *b as usize).sum(),
                validity,
            )
            .map(|a| a.into_array())
        }
        DType::List(..) | DType::FixedSizeList(..) => {
            let mut indices = Vec::new();
            for (idx, bool) in filter.iter().enumerate() {
                if *bool {
                    indices.push(idx);
                }
            }
            take_canonical_array_non_nullable_indices(array, indices.as_slice())
        }
        d @ (DType::Null | DType::Extension(_) | DType::Variant(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}
