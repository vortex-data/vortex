use num_traits::NumCast;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::{NativePType, Nullability, PType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
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

        let nullability = lhs.dtype.nullability() | rhs.dtype().nullability();
        let scalar_type = lhs.msp.dtype().with_nullability(nullability);

        let rhs_decimal = rhs_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");
        let Some(encoded_scalar) =
            decimal_value_wrapper_to_primitive(rhs_decimal, lhs.msp.as_primitive_typed().ptype())
                .map(|value| Scalar::new(scalar_type.clone(), value))
        else {
            // here the scalar value is bigger than the msp type.
            // TODO(joe): fixme, when allowing lsp values.
            return Ok(Some(
                ConstantArray::new(unconvertible_value(operator, nullability), lhs.len())
                    .to_array(),
            ));
        };
        let encoded_const = ConstantArray::new(encoded_scalar, rhs.len());
        compare(&lhs.msp, &encoded_const.to_array(), operator).map(Some)
    }
}

fn unconvertible_value(operator: Operator, nullability: Nullability) -> Scalar {
    // v op unconvertible where unconvertible > v_max
    match operator {
        // v is never eq or gt/gte
        Operator::Eq | Operator::Gt | Operator::Gte => Scalar::bool(false, nullability),
        // v is always eq or gt/gte
        Operator::NotEq | Operator::Lt | Operator::Lte => Scalar::bool(true, nullability),
    }
}

// this value return None is the decimal scalar cannot be cast the ptype.
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
    match_each_decimal_value!(decimal_value, |decimal_v| {
        Some(ScalarValue::from(<P as NumCast>::from(decimal_v)?))
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

    #[test]
    fn compare_decimal_const_unconvertible_comparison() {
        let decimal_dtype = DecimalDType::new(40, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).to_array(),
            vec![],
            decimal_dtype,
        )
        .unwrap()
        .to_array();
        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::new(dtype, DecimalValue::I128(-9999999999999965304).into()),
            lhs.len(),
        );

        let res = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        assert_eq!(
            res.to_bool().unwrap().bool_vec().unwrap(),
            vec![false, false, false]
        );

        let res = compare(lhs.as_ref(), rhs.as_ref(), Operator::Gt).unwrap();
        assert_eq!(
            res.to_bool().unwrap().bool_vec().unwrap(),
            vec![false, false, false]
        );

        let res = compare(lhs.as_ref(), rhs.as_ref(), Operator::Lt).unwrap();
        assert_eq!(
            res.to_bool().unwrap().bool_vec().unwrap(),
            vec![true, true, true]
        );
    }
}
