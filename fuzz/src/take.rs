use arrow_buffer::ArrowNativeType;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::builders::{builder_with_capacity, ArrayBuilderExt};
use vortex_array::compute::scalar_at;
use vortex_array::validity::Validity;
use vortex_array::variants::StructArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::VortexResult;

pub fn take_canonical_array(array: &dyn Array, indices: &[usize]) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let validity_idx = array
            .validity_mask()?
            .to_boolean_buffer()
            .iter()
            .collect::<Vec<_>>();

        Validity::from_iter(indices.iter().map(|i| validity_idx[*i]))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool()?;
            let vec_values = bool_array.boolean_buffer().iter().collect::<Vec<_>>();
            Ok(
                BoolArray::new(indices.iter().map(|i| vec_values[*i]).collect(), validity)
                    .into_array(),
            )
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.to_primitive()?;
            match_each_native_ptype!(p, |$P| {
                Ok(take_primitive::<$P>(primitive_array, validity, indices))
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview()?;
            let values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())?;
            Ok(VarBinViewArray::from_iter(
                indices.iter().map(|i| values[*i].clone()),
                array.dtype().clone(),
            )
            .into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.to_struct()?;
            let taken_children = struct_array
                .fields()
                .iter()
                .map(|c| take_canonical_array(c, indices))
                .collect::<VortexResult<Vec<_>>>()?;

            StructArray::try_new(
                struct_array.names().clone(),
                taken_children,
                indices.len(),
                validity,
            )
            .map(|a| a.into_array())
        }
        DType::List(..) => {
            let mut builder = builder_with_capacity(array.dtype(), indices.len());
            for idx in indices {
                builder.append_scalar(&scalar_at(array, *idx)?)?;
            }
            Ok(builder.finish())
        }
        _ => unreachable!("Not a canonical array"),
    }
}

fn take_primitive<T: NativePType + ArrowNativeType>(
    primitive_array: PrimitiveArray,
    validity: Validity,
    indices: &[usize],
) -> ArrayRef {
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
