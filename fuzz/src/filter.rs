use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{
    BoolArray, BooleanBuffer, PrimitiveArray, StructArray, VarBinViewArray,
};
use vortex_array::validity::Validity;
use vortex_array::variants::StructArrayTrait;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::VortexResult;

use crate::take::take_canonical_array;

pub fn filter_canonical_array(array: &dyn Array, filter: &[bool]) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let validity_buff = array.validity_mask()?.into_array().to_bool()?;
        Validity::from_iter(
            filter
                .iter()
                .zip(validity_buff.boolean_buffer())
                .filter(|(f, _)| **f)
                .map(|(_, v)| v),
        )
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool()?;
            Ok(BoolArray::new(
                BooleanBuffer::from_iter(
                    filter
                        .iter()
                        .zip(bool_array.boolean_buffer().iter())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v),
                ),
                validity,
            )
            .into_array())
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |$P| {
            let primitive_array = array.to_primitive()?;
            Ok(PrimitiveArray::new(
                filter
                    .iter()
                    .zip(primitive_array.as_slice::<$P>().iter().copied())
                    .filter(|(f, _)| **f)
                    .map(|(_, v)| v)
                    .collect::<Buffer<_>>(),
                validity,
            )
            .into_array())
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview()?;
            let values = utf8.with_iterator(|iter| {
                iter.zip(filter.iter())
                    .filter(|(_, f)| **f)
                    .map(|(v, _)| v.map(|u| u.to_vec()))
                    .collect::<Vec<_>>()
            })?;
            Ok(VarBinViewArray::from_iter(values, array.dtype().clone()).into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.to_struct()?;
            let filtered_children = struct_array
                .fields()
                .iter()
                .map(|c| filter_canonical_array(c, filter))
                .collect::<VortexResult<Vec<_>>>()?;

            StructArray::try_new(
                struct_array.names().clone(),
                filtered_children,
                filter.iter().filter(|b| **b).map(|b| *b as usize).sum(),
                validity,
            )
            .map(|a| a.into_array())
        }
        DType::List(..) => {
            let mut indices = Vec::new();
            for (idx, bool) in filter.iter().enumerate() {
                if *bool {
                    indices.push(idx);
                }
            }
            take_canonical_array(array, &indices)
        }
        _ => unreachable!("Not a canonical array"),
    }
}
