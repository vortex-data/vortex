use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::VortexExpect;

pub fn slice_canonical_array(array: &ArrayData, start: usize, stop: usize) -> ArrayData {
    let validity = if array.dtype().is_nullable() {
        let bool_buff = array
            .logical_validity()
            .into_array()
            .into_bool()
            .unwrap()
            .boolean_buffer();

        Validity::from(bool_buff.slice(start, stop - start))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().into_bool().unwrap();
            let sliced_bools = bool_array.boolean_buffer().slice(start, stop - start);
            BoolArray::try_new(sliced_bools, validity)
                .vortex_expect("Validity length cannot mismatch")
                .into_array()
        }
        DType::Primitive(p, _) => match_each_native_ptype!(p, |$P| {
            let primitive_array = array.clone().into_primitive().unwrap();
            let vec_values = primitive_array.into_maybe_null_slice::<$P>();
            PrimitiveArray::from_vec(vec_values[start..stop].into(), validity).into_array()
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().into_varbinview().unwrap();
            let values = utf8
                .with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())
                .unwrap();
            VarBinViewArray::from_iter(values[start..stop].iter().cloned(), array.dtype().clone())
                .into_array()
        }
        DType::Struct(..) => {
            let struct_array = array.clone().into_struct().unwrap();
            let sliced_children = struct_array
                .children()
                .map(|c| slice_canonical_array(&c, start, stop))
                .collect::<Vec<_>>();
            StructArray::try_new(
                struct_array.names().clone(),
                sliced_children,
                stop - start,
                validity,
            )
            .unwrap()
            .into_array()
        }
        _ => unreachable!("Not a canonical array"),
    }
}
