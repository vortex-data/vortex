use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_native_ptype, DType};

pub fn filter_canonical_array(array: &ArrayData, filter: &[bool]) -> ArrayData {
    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().into_bool().unwrap();
            let vec_validity = bool_array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer();
            BoolArray::from_vec(
                filter
                    .iter()
                    .zip(bool_array.boolean_buffer().iter())
                    .filter(|(f, _)| **f)
                    .map(|(_, v)| v)
                    .collect::<Vec<_>>(),
                Validity::from(
                    filter
                        .iter()
                        .zip(vec_validity.iter())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v)
                        .collect::<Vec<_>>(),
                ),
            )
            .into_array()
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |$P| {
            let primitive_array = array.clone().into_primitive().unwrap();
            let vec_validity = primitive_array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer();
            PrimitiveArray::from_vec(
                filter
                    .iter()
                    .zip(primitive_array.maybe_null_slice::<$P>().iter().copied())
                    .filter(|(f, _)| **f)
                    .map(|(_, v)| v)
                    .collect::<Vec<_>>(),
                Validity::from(
                    filter
                        .iter()
                        .zip(vec_validity.iter())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v)
                        .collect::<Vec<_>>(),
                ),
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
            let vec_validity = struct_array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer();

            StructArray::try_new(
                struct_array.names().clone(),
                filtered_children,
                filter.iter().filter(|b| **b).map(|b| *b as usize).sum(),
                Validity::from(
                    filter
                        .iter()
                        .zip(vec_validity.iter())
                        .filter(|(f, _)| **f)
                        .map(|(_, v)| v)
                        .collect::<Vec<_>>(),
                ),
            )
            .unwrap()
            .into_array()
        }
        _ => unreachable!("Not a canonical array"),
    }
}
