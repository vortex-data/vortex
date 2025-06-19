use arrow_buffer::ArrowNativeType;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{BoolArray, DecimalArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::builders::{ArrayBuilderExt, builder_with_capacity};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, DecimalDType, NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_scalar::{NativeDecimalType, match_each_decimal_value_type};

pub fn take_canonical_array(array: &dyn Array, indices: &[usize]) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let validity_idx = array.validity_mask()?.to_boolean_buffer();

        Validity::from_iter(indices.iter().map(|i| validity_idx.value(*i)))
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
            match_each_native_ptype!(p, |P| {
                Ok(take_primitive::<P>(primitive_array, validity, indices))
            })
        }
        DType::Decimal(d, _) => {
            let decimal_array = array.to_decimal()?;

            match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                Ok(take_decimal::<D>(decimal_array, d, validity, indices))
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
                builder.append_scalar(&array.scalar_at(*idx)?)?;
            }
            Ok(builder.finish())
        }
        d @ (DType::Null | DType::Extension(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
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

fn take_decimal<D: NativeDecimalType>(
    array: DecimalArray,
    decimal_type: &DecimalDType,
    validity: Validity,
    indices: &[usize],
) -> ArrayRef {
    let buf = array.buffer::<D>();
    let vec_values = buf.as_slice();
    DecimalArray::new(
        indices
            .iter()
            .map(|i| vec_values[*i])
            .collect::<Buffer<D>>(),
        *decimal_type,
        validity,
    )
    .into_array()
}
