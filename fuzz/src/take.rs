use arrow_buffer::ArrowNativeType;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::builders::{builder_with_capacity, ArrayBuilderExt};
use vortex_array::compute::scalar_at;
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
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
        DType::Primitive(p, _) => {
            let primitive_array = array.clone().into_primitive().unwrap();
            match_each_native_ptype!(p, |$P| {
                take_primitive::<$P>(primitive_array, validity, indices)
            })
        }
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
        DType::List(..) => {
            let mut builder = builder_with_capacity(array.dtype(), indices.len());
            for idx in indices {
                builder
                    .append_scalar(&scalar_at(array, *idx).unwrap())
                    .unwrap();
            }
            builder.finish().unwrap()
        }
        _ => unreachable!("Not a canonical array"),
    }
}

fn take_primitive<T: NativePType + ArrowNativeType>(
    primitive_array: PrimitiveArray,
    validity: Validity,
    indices: &[usize],
) -> ArrayData {
    let vec_values = primitive_array.as_slice::<T>().to_vec();
    PrimitiveArray::new(
        indices
            .iter()
            .map(|i| vec_values[*i])
            .collect::<Buffer<T>>(),
        validity,
    )
    .into_array()
}
