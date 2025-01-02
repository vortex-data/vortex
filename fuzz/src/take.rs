use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::VortexExpect;

pub fn take_canonical_array(array: &ArrayData, indices: &[usize]) -> ArrayData {
    let validity = if array.dtype().is_nullable() {
        let validity_idx = array
            .logical_validity()
            .into_array()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .iter()
            .collect::<Vec<_>>();

        Validity::from_iter(indices.iter().map(|i| validity_idx[*i]))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().into_bool().unwrap();
            let vec_values = bool_array.boolean_buffer().iter().collect::<Vec<_>>();
            BoolArray::try_new(indices.iter().map(|i| vec_values[*i]).collect(), validity)
                .vortex_expect("Validity length cannot mismatch")
                .into_array()
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |$P| {
            let primitive_array = array.clone().into_primitive().unwrap();
            let vec_values = primitive_array
                .as_slice::<$P>()
                .iter()
                .copied()
                .collect::<Vec<_>>();
            PrimitiveArray::new(indices.iter().map(|i| vec_values[*i]).collect::<Buffer<$P>>(), validity)
                .into_array()
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().into_varbinview().unwrap();
            let values = utf8
                .with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())
                .unwrap();
            VarBinViewArray::from_iter(
                indices.iter().map(|i| values[*i].clone()),
                array.dtype().clone(),
            )
            .into_array()
        }
        DType::Struct(..) => {
            let struct_array = array.clone().into_struct().unwrap();
            let taken_children = struct_array
                .children()
                .map(|c| take_canonical_array(&c, indices))
                .collect::<Vec<_>>();

            StructArray::try_new(
                struct_array.names().clone(),
                taken_children,
                indices.len(),
                validity,
            )
            .unwrap()
            .into_array()
        }
        _ => unreachable!("Not a canonical array"),
    }
}
