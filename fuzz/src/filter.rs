use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, BooleanBuffer, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::VortexExpect;

pub fn filter_canonical_array(array: &ArrayData, filter: &[bool]) -> ArrayData {
    let validity = if array.dtype().is_nullable() {
        let validity_buff = array
            .logical_validity()
            .into_array()
            .into_bool()
            .unwrap()
            .boolean_buffer();
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
            let bool_array = array.clone().into_bool().unwrap();
            BoolArray::try_new(
                BooleanBuffer::from_iter(
                    filter
                        .iter()
                        .zip(bool_array.boolean_buffer().iter())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v),
                ),
                validity,
            )
            .vortex_expect("Validity length cannot mismatch")
            .into_array()
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |$P| {
            let primitive_array = array.clone().into_primitive().unwrap();
            PrimitiveArray::new(
                filter
                    .iter()
                    .zip(primitive_array.as_slice::<$P>().iter().copied())
                    .filter(|(f, _)| **f)
                    .map(|(_, v)| v)
                    .collect::<Buffer<_>>(),
                validity,
            )
            .into_array()
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().into_varbinview().unwrap();
            let values = utf8
                .with_iterator(|iter| {
                    iter.zip(filter.iter())
                        .filter(|(_, f)| **f)
                        .map(|(v, _)| v.map(|u| u.to_vec()))
                        .collect::<Vec<_>>()
                })
                .unwrap();
            VarBinViewArray::from_iter(values, array.dtype().clone()).into_array()
        }
        DType::Struct(..) => {
            let struct_array = array.clone().into_struct().unwrap();
            let filtered_children = struct_array
                .children()
                .map(|c| filter_canonical_array(&c, filter))
                .collect::<Vec<_>>();

            StructArray::try_new(
                struct_array.names().clone(),
                filtered_children,
                filter.iter().filter(|b| **b).map(|b| *b as usize).sum(),
                validity,
            )
            .unwrap()
            .into_array()
        }
        _ => unreachable!("Not a canonical array"),
    }
}
