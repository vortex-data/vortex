use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, BooleanBuffer, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::VortexExpect;

pub fn slice_canonical_array(array: &ArrayData, start: usize, stop: usize) -> ArrayData {
    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().into_bool().unwrap();
            let vec_values = bool_array.boolean_buffer().iter().collect::<Vec<_>>();
            let vec_validity = bool_array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>();
            BoolArray::try_new(
                BooleanBuffer::from(&vec_values[start..stop]),
                Validity::from_iter(vec_validity[start..stop].iter().copied()),
            )
            .vortex_expect("Validity length cannot mismatch")
            .into_array()
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |$P| {
            let primitive_array = array.clone().into_primitive().unwrap();
            let vec_values = primitive_array
                .maybe_null_slice::<$P>()
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let vec_validity = primitive_array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>();
            PrimitiveArray::from_vec(
                Vec::from(&vec_values[start..stop]),
                Validity::from_iter(vec_validity[start..stop].iter().cloned()),
            )
            .into_array()
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().into_varbinview().unwrap();
            let values = utf8
                .with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())
                .unwrap();
            VarBinViewArray::from_iter(Vec::from(&values[start..stop]), array.dtype().clone())
                .into_array()
        }
        DType::Struct(..) => {
            let struct_array = array.clone().into_struct().unwrap();
            let sliced_children = struct_array
                .children()
                .map(|c| slice_canonical_array(&c, start, stop))
                .collect::<Vec<_>>();
            let vec_validity = struct_array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>();

            StructArray::try_new(
                struct_array.names().clone(),
                sliced_children,
                stop - start,
                Validity::from_iter(vec_validity[start..stop].iter().cloned()),
            )
            .unwrap()
            .into_array()
        }
        _ => unreachable!("Not a canonical array"),
    }
}
