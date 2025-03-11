use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::compute::{Operator, scalar_at, scalar_cmp};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

pub fn compare_canonical_array(
    array: &dyn Array,
    value: &Scalar,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    match array.dtype() {
        DType::Bool(_) => {
            let bool = value.as_bool().value();
            Ok(compare_to(
                array
                    .to_bool()?
                    .boolean_buffer()
                    .iter()
                    .zip(array.validity_mask()?.to_boolean_buffer().iter())
                    .map(|(b, v)| v.then_some(b)),
                bool,
                operator,
            ))
        }
        DType::Primitive(p, _) => {
            let primitive = value.as_primitive();
            let primitive_array = array.to_primitive()?;
            match_each_native_ptype!(p, |$P| {
                let pval = primitive.typed_value::<$P>();
                Ok(compare_to(
                    primitive_array
                        .as_slice::<$P>()
                        .iter()
                        .copied()
                        .zip(array.validity_mask()?.to_boolean_buffer().iter())
                        .map(|(b, v)| v.then_some(b)),
                    pval,
                    operator,
                ))
            })
        }
        DType::Utf8(_) => array.to_varbinview()?.with_iterator(|iter| {
            let utf8_value = value.as_utf8().value();
            compare_to(
                iter.map(|v| v.map(|b| unsafe { str::from_utf8_unchecked(b) })),
                utf8_value.as_deref(),
                operator,
            )
        }),
        DType::Binary(_) => array.to_varbinview()?.with_iterator(|iter| {
            let binary_value = value.as_binary().value();
            compare_to(
                // Don't understand the lifetime problem here but identity map makes it go away
                #[allow(clippy::map_identity)]
                iter.map(|v| v),
                binary_value.as_deref(),
                operator,
            )
        }),
        DType::Struct(..) | DType::List(..) => {
            let scalar_vals = (0..array.len())
                .map(|i| scalar_at(array, i))
                .collect::<VortexResult<Vec<_>>>()?;
            Ok(BoolArray::from_iter(
                scalar_vals
                    .iter()
                    .map(|v| scalar_cmp(v, value, operator).as_bool().value()),
            )
            .into_array())
        }
        d => unreachable!("DType {d} not supported for fuzzing"),
    }
}

fn compare_to<T: PartialOrd + PartialEq>(
    values: impl Iterator<Item = Option<T>>,
    value: Option<T>,
    operator: Operator,
) -> ArrayRef {
    BoolArray::from_iter(values.map(|v| match operator {
        Operator::Eq => v == value,
        Operator::NotEq => v != value,
        Operator::Gt => v > value,
        Operator::Gte => v >= value,
        Operator::Lt => v < value,
        Operator::Lte => v <= value,
    }))
    .into_array()
}
