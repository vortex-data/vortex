use num_traits::NumCast;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::{NativePType, PType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_scalar::{DecimalValue, Scalar, ScalarValue, match_each_decimal_value};

use crate::DecimalBytePartsVTable;

impl CompareKernel for DecimalBytePartsVTable {
    fn compare(
        &self,
        lhs: &Self::Array,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(joe): support compare with lower parts
        if !lhs.lower_parts.is_empty() {
            return Ok(None);
        }
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        let scalar_type = lhs
            .msp
            .dtype()
            .with_nullability(lhs.dtype.nullability() | rhs.dtype().nullability());

        let encoded_scalar = rhs_const
            .as_decimal()
            .decimal_value()
            .and_then(|value| {
                decimal_value_wrapper_to_primitive(value, lhs.msp.as_primitive_typed().ptype())
            })
            .map(|value| Scalar::new(scalar_type.clone(), value))
            .unwrap_or_else(|| Scalar::null(scalar_type));
        let encoded_const = ConstantArray::new(encoded_scalar, rhs.len());
        compare(&lhs.msp, &encoded_const.to_array(), operator).map(Some)
    }
}

// clippy prefers smaller functions
fn decimal_value_wrapper_to_primitive(
    decimal_value: DecimalValue,
    ptype: PType,
) -> Option<ScalarValue> {
    match_each_integer_ptype!(ptype, |P| {
        decimal_value_to_primitive::<P>(decimal_value)
    })
}

fn decimal_value_to_primitive<P>(decimal_value: DecimalValue) -> Option<ScalarValue>
where
    P: NativePType + NumCast,
    ScalarValue: From<P>,
{
    match_each_decimal_value!(decimal_value, |$decimal_v| {
        Some(ScalarValue::from(<P as NumCast>::from($decimal_v)?))
    })
}

register_kernel!(CompareKernelAdapter(DecimalBytePartsVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{Operator, compare};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::DecimalBytePartsArray;

    #[test]
    fn compare_decimal_const() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).to_array(),
            vec![],
            decimal_dtype,
        )
        .unwrap()
        .to_array();
        let rhs = ConstantArray::new(Scalar::new(dtype, DecimalValue::I64(400).into()), lhs.len());

        let res = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        assert_eq!(
            res.to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![false, false, true]
        );
    }
}
