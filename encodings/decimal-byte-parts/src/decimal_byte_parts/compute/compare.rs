// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use Sign::Negative;
use num_traits::{Bounded, NumCast};
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::{NativePType, Nullability, PType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{DecimalValue, Scalar, ScalarValue, ToPrimitive, match_each_decimal_value};

use crate::DecimalBytePartsVTable;
use crate::decimal_byte_parts::compute::compare::Sign::Positive;

impl CompareKernel for DecimalBytePartsVTable {
    fn compare(
        &self,
        lhs: &Self::Array,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        let nullability = lhs.dtype.nullability() | rhs.dtype().nullability();
        let scalar_type = lhs.msp.dtype().with_nullability(nullability);

        let rhs_decimal = rhs_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");

        match decimal_value_wrapper_to_primitive(rhs_decimal, lhs.msp.as_primitive_typed().ptype())
            .map(|value| Scalar::new(scalar_type.clone(), value))
        {
            Ok(encoded_scalar) => {
                let encoded_const = ConstantArray::new(encoded_scalar, rhs.len());
                compare(&lhs.msp, &encoded_const.to_array(), operator).map(Some)
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
                        .to_array(),
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

fn unconvertible_value(sign: Sign, operator: Operator, nullability: Nullability) -> Scalar {
    match operator {
        Operator::Eq => Scalar::bool(false, nullability),
        Operator::NotEq => Scalar::bool(true, nullability),
        Operator::Gt | Operator::Gte => Scalar::bool(matches!(sign, Negative), nullability),
        Operator::Lt | Operator::Lte => Scalar::bool(matches!(sign, Positive), nullability),
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
    P: NativePType + NumCast + Bounded + ToPrimitive,
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

register_kernel!(CompareKernelAdapter(DecimalBytePartsVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, ConstantArray, PrimitiveArray};
    use vortex_array::compute::{Operator, compare};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, DecimalDType, Nullability};
    use vortex_error::VortexResult;
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::DecimalBytePartsArray;

    #[test]
    fn compare_decimal_const() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).to_array(),
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

        let res = compare(lhs.as_ref(), rhs.as_ref(), Operator::Lte)?;
        assert_eq!(res.scalar_at(0).as_bool().value(), None);
        assert_eq!(res.scalar_at(1).as_bool().value(), Some(true));
        assert_eq!(res.scalar_at(2).as_bool().value(), Some(true));
        assert_eq!(res.scalar_at(3).as_bool().value(), Some(true));

        Ok(())
    }

    #[test]
    fn compare_decimal_const_unconvertible_comparison() {
        let decimal_dtype = DecimalDType::new(40, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).to_array(),
            decimal_dtype,
        )
        .unwrap()
        .to_array();
        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::new(
                dtype.clone(),
                DecimalValue::I128(-9999999999999965304).into(),
            ),
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
            vec![true, true, true]
        );

        let res = compare(lhs.as_ref(), rhs.as_ref(), Operator::Lt).unwrap();
        assert_eq!(
            res.to_bool().unwrap().bool_vec().unwrap(),
            vec![false, false, false]
        );

        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::new(dtype, DecimalValue::I128(9999999999999965304).into()),
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
