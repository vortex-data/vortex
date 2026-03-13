// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use Sign::Negative;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::ToI256;
use vortex_array::match_each_decimal_value;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::compute::compare::Sign::Positive;

impl CompareKernel for DecimalByteParts {
    fn compare(
        lhs: &Self::Array,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        let nullability = lhs.dtype.nullability() | rhs.dtype().nullability();
        let scalar_type = lhs.msp().dtype().with_nullability(nullability);

        let rhs_decimal = rhs_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");

        match decimal_value_wrapper_to_primitive(
            rhs_decimal,
            lhs.msp().as_primitive_typed().ptype(),
        ) {
            Ok(value) => {
                let encoded_scalar = Scalar::try_new(scalar_type, Some(value))?;
                let encoded_const = ConstantArray::new(encoded_scalar, rhs.len());
                lhs.msp()
                    .binary(encoded_const.into_array(), Operator::from(operator))
                    .map(Some)
            }

            Err(sign) => {
                // If the MSP and the constant are non-null, we know that failing to coerce the
                // constant into the MSP bit-width means that it is larger/smaller
                // (depending on the `sign`) than all values in MSP.
                // If the LHS or the RHS contain nulls, then we must fallback to the canonicalized
                // implementation which does null-checking instead.
                if lhs.all_valid()? && rhs.all_valid()? {
                    Ok(Some(
                        ConstantArray::new(
                            unconvertible_value(sign, operator, nullability),
                            lhs.len(),
                        )
                        .into_array(),
                    ))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

// Used to represent the overflow direction when trying to
// convert into the scalar type.
#[derive(Debug)]
enum Sign {
    Positive,
    Negative,
}

fn unconvertible_value(sign: Sign, operator: CompareOperator, nullability: Nullability) -> Scalar {
    match operator {
        CompareOperator::Eq => Scalar::bool(false, nullability),
        CompareOperator::NotEq => Scalar::bool(true, nullability),
        CompareOperator::Gt | CompareOperator::Gte => {
            Scalar::bool(matches!(sign, Negative), nullability)
        }
        CompareOperator::Lt | CompareOperator::Lte => {
            Scalar::bool(matches!(sign, Positive), nullability)
        }
    }
}

// this value return None is the decimal scalar cannot be cast the ptype.
fn decimal_value_wrapper_to_primitive(
    decimal_value: DecimalValue,
    ptype: PType,
) -> Result<ScalarValue, Sign> {
    match_each_integer_ptype!(ptype, |P| {
        decimal_value_to_primitive::<P>(decimal_value)
    })
}

fn decimal_value_to_primitive<P>(decimal_value: DecimalValue) -> Result<ScalarValue, Sign>
where
    P: IntegerPType + ToI256,
    ScalarValue: From<P>,
{
    match_each_decimal_value!(decimal_value, |decimal_v| {
        let Some(encoded) = <P as NumCast>::from(decimal_v) else {
            let decimal_i256 = decimal_v
                .to_i256()
                .vortex_expect("i256 is big enough for any DecimalValue");
            return if decimal_i256
                > P::max_value()
                    .to_i256()
                    .vortex_expect("i256 is big enough for any PType")
            {
                Err(Positive)
            } else {
                assert!(
                    decimal_i256
                        < P::min_value()
                            .to_i256()
                            .vortex_expect("i256 is big enough for any PType")
                );
                Err(Negative)
            };
        };
        Ok(ScalarValue::from(encoded))
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::DecimalBytePartsArray;

    #[test]
    fn compare_decimal_const() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).into_array(),
            decimal_dtype,
        )
        .unwrap()
        .into_array();
        let rhs = ConstantArray::new(
            Scalar::try_new(dtype, Some(DecimalValue::I64(400).into())).unwrap(),
            lhs.len(),
        );

        let res = lhs.binary(rhs.into_array(), Operator::Eq).unwrap();

        let expected = BoolArray::from_iter([Some(false), Some(false), Some(true)]).into_array();
        assert_arrays_eq!(res, expected);
    }

    #[test]
    fn test_byteparts_compare_nullable() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(19, -11);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(
                buffer![1i64, 2i64, 3i64, 4i64],
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            )
            .into_array(),
            decimal_type,
        )?;

        let rhs = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(289888198),
                decimal_type,
                Nullability::NonNullable,
            ),
            4,
        )
        .into_array();

        let res = lhs.into_array().binary(rhs, Operator::Lte)?;
        let expected =
            BoolArray::from_iter([None, Some(true), Some(true), Some(true)]).into_array();
        assert_arrays_eq!(res, expected);

        Ok(())
    }

    #[test]
    fn compare_decimal_const_unconvertible_comparison() {
        let decimal_dtype = DecimalDType::new(40, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).into_array(),
            decimal_dtype,
        )
        .unwrap()
        .into_array();
        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::try_new(
                dtype.clone(),
                Some(DecimalValue::I128(-9999999999999965304).into()),
            )
            .unwrap(),
            lhs.len(),
        );

        let res = lhs.binary(rhs.clone().into_array(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected);

        let res = lhs.binary(rhs.clone().into_array(), Operator::Gt).unwrap();
        let expected = BoolArray::from_iter([Some(true), Some(true), Some(true)]).into_array();
        assert_arrays_eq!(res, expected);

        let res = lhs.binary(rhs.into_array(), Operator::Lt).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected);

        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::try_new(dtype, Some(DecimalValue::I128(9999999999999965304).into())).unwrap(),
            lhs.len(),
        );

        let res = lhs.binary(rhs.clone().into_array(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected);

        let res = lhs.binary(rhs.clone().into_array(), Operator::Gt).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected);

        let res = lhs.binary(rhs.into_array(), Operator::Lt).unwrap();
        let expected = BoolArray::from_iter([Some(true), Some(true), Some(true)]).into_array();
        assert_arrays_eq!(res, expected);
    }
}
