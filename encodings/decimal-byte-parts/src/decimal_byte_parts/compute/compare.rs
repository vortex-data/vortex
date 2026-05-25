// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use Sign::Negative;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
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
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::compute::compare::Sign::Positive;

impl CompareKernel for DecimalByteParts {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Multi-part decimals: compare limb-wise against another byte-parts column of the same
        // layout, without ever reconstructing the wide (i128/i256) value. This is the path that
        // beats the canonical route, which must first decode every value back to i128/i256.
        if lhs.num_lower_parts() > 0 {
            return lexicographic_compare(lhs, rhs, operator, ctx);
        }

        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
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
                if lhs.array().all_valid(ctx)? && rhs.all_valid(ctx)? {
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

/// Compares two byte-parts decimal columns of the same layout limb-by-limb, most-significant part
/// first. Because a decimal stored as `(msp: i64, lower: u64, ...)` is exactly the two's-complement
/// limb decomposition of the wide value, lexicographic comparison (signed on the msp, unsigned on
/// the lower limbs) reproduces the numeric ordering — without materializing the i128/i256 value.
///
/// Returns `Ok(None)` (so the engine falls back to canonicalization) when `rhs` is not a byte-parts
/// column with the standard `i64 + u64*` layout matching `lhs`.
fn lexicographic_compare(
    lhs: ArrayView<'_, DecimalByteParts>,
    rhs: &ArrayRef,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(rhs) = rhs.as_opt::<DecimalByteParts>() else {
        return Ok(None);
    };
    let k = lhs.num_lower_parts();
    if rhs.num_lower_parts() != k || !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
        return Ok(None);
    }
    // The native path only supports the canonical wide layout: signed i64 msp + unsigned u64 limbs.
    let is_i64 = |a: &ArrayRef| PType::try_from(a.dtype()).ok() == Some(PType::I64);
    let is_u64 = |a: &ArrayRef| PType::try_from(a.dtype()).ok() == Some(PType::U64);
    if !is_i64(lhs.msp()) || !is_i64(rhs.msp()) {
        return Ok(None);
    }
    if (0..k).any(|i| !is_u64(lhs.lower_part(i)) || !is_u64(rhs.lower_part(i))) {
        return Ok(None);
    }

    let lhs_msp = lhs.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let rhs_msp = rhs.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let n = lhs_msp.len();

    let lhs_lower = (0..k)
        .map(|i| lhs.lower_part(i).clone().execute::<PrimitiveArray>(ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let rhs_lower = (0..k)
        .map(|i| rhs.lower_part(i).clone().execute::<PrimitiveArray>(ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    let a0 = lhs_msp.as_slice::<i64>();
    let b0 = rhs_msp.as_slice::<i64>();
    let lo_a: Vec<&[u64]> = lhs_lower.iter().map(|p| p.as_slice::<u64>()).collect();
    let lo_b: Vec<&[u64]> = rhs_lower.iter().map(|p| p.as_slice::<u64>()).collect();

    // Row-major lexicographic compare with early-exit: the most significant limb is signed, the rest
    // unsigned. Per-row ordering is computed with the running state in registers and emitted in a
    // single pass, so no full value is ever reconstructed.
    let bits = BitBuffer::from_iter((0..n).map(|i| {
        let mut ord = a0[i].cmp(&b0[i]);
        let mut limb = 0;
        while ord == Ordering::Equal && limb < k {
            ord = lo_a[limb][i].cmp(&lo_b[limb][i]);
            limb += 1;
        }
        match operator {
            CompareOperator::Eq => ord == Ordering::Equal,
            CompareOperator::NotEq => ord != Ordering::Equal,
            CompareOperator::Lt => ord == Ordering::Less,
            CompareOperator::Lte => ord != Ordering::Greater,
            CompareOperator::Gt => ord == Ordering::Greater,
            CompareOperator::Gte => ord != Ordering::Less,
        }
    }));

    // A row is null iff either operand is null; both carry their validity in the msp.
    let validity = lhs_msp.validity()?.and(rhs_msp.validity()?)?;
    Ok(Some(BoolArray::new(bits, validity).into_array()))
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::DecimalByteParts;

    #[test]
    fn compare_decimal_const() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalByteParts::try_new(
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
        let lhs = DecimalByteParts::try_new(
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
        let lhs = DecimalByteParts::try_new(
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

    /// Builds a multi-part byte-parts column and an equivalent canonical decimal column from the
    /// same optional `i128` values, so the native limb-wise compare can be checked against the
    /// canonical (Arrow) compare.
    fn i128_pair(values: &[Option<i128>], dtype: DecimalDType) -> (ArrayRef, ArrayRef) {
        let validity =
            Validity::Array(BoolArray::from_iter(values.iter().map(Option::is_some)).into_array());
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|v| (v.unwrap_or(0) >> 64) as i64)
                .collect::<Buffer<i64>>(),
            validity,
        );
        let low = PrimitiveArray::new(
            values
                .iter()
                .map(|v| v.unwrap_or(0) as u64)
                .collect::<Buffer<u64>>(),
            Validity::NonNullable,
        );
        let bp = DecimalByteParts::try_new_parts(msp.into_array(), vec![low.into_array()], dtype)
            .unwrap()
            .into_array();
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        (bp, canon)
    }

    fn i256_pair(values: &[Option<i256>], dtype: DecimalDType) -> (ArrayRef, ArrayRef) {
        let validity =
            Validity::Array(BoolArray::from_iter(values.iter().map(Option::is_some)).into_array());
        let z = i256::ZERO;
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|v| (v.unwrap_or(z).to_parts().1 >> 64) as i64)
                .collect::<Buffer<i64>>(),
            validity,
        );
        let mk = |f: fn(i256) -> u64| {
            PrimitiveArray::new(
                values
                    .iter()
                    .map(|v| f(v.unwrap_or(z)))
                    .collect::<Buffer<u64>>(),
                Validity::NonNullable,
            )
            .into_array()
        };
        let lowers = vec![
            mk(|v| v.to_parts().1 as u64),
            mk(|v| (v.to_parts().0 >> 64) as u64),
            mk(|v| v.to_parts().0 as u64),
        ];
        let bp = DecimalByteParts::try_new_parts(msp.into_array(), lowers, dtype)
            .unwrap()
            .into_array();
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        (bp, canon)
    }

    const OPERATORS: [Operator; 6] = [
        Operator::Eq,
        Operator::NotEq,
        Operator::Lt,
        Operator::Lte,
        Operator::Gt,
        Operator::Gte,
    ];

    #[rstest]
    fn native_i128_matches_canonical(
        #[values(0, 1, 2, 3, 4, 5)] op_idx: usize,
    ) -> VortexResult<()> {
        let operator = OPERATORS[op_idx];
        let dtype = DecimalDType::new(38, 2);
        let p = 10i128.pow(30);
        // Equal values, differing only in the low limb, differing in the high limb, negatives,
        // sign-crossing, and nulls — to exercise every branch of the lexicographic logic.
        let a = [
            Some(p),
            Some(p + 1),
            Some(p),
            Some(-p),
            None,
            Some(5),
            Some(-p),
            Some(p),
        ];
        let b = [
            Some(p),
            Some(p),
            Some(p + 1),
            Some(p),
            Some(7),
            None,
            Some(-p - 1),
            Some(p),
        ];
        let (bp_a, canon_a) = i128_pair(&a, dtype);
        let (bp_b, canon_b) = i128_pair(&b, dtype);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let native = bp_a
            .binary(bp_b, operator)?
            .execute::<BoolArray>(&mut ctx)?;
        let canonical = canon_a
            .binary(canon_b, operator)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(native.into_array(), canonical.into_array());
        Ok(())
    }

    #[rstest]
    fn native_i256_matches_canonical(
        #[values(0, 1, 2, 3, 4, 5)] op_idx: usize,
    ) -> VortexResult<()> {
        let operator = OPERATORS[op_idx];
        let dtype = DecimalDType::new(60, 2);
        let hi = i256::from_parts(0, 10i128.pow(30));
        let a = [
            Some(hi),
            Some(hi + i256::ONE),
            Some(hi),
            Some(-hi),
            None,
            Some(i256::from_i128(5)),
            Some(i256::from_parts(u128::MAX, 1)),
        ];
        let b = [
            Some(hi),
            Some(hi),
            Some(hi + i256::ONE),
            Some(hi),
            Some(i256::from_i128(7)),
            None,
            Some(i256::from_parts(0, 1)),
        ];
        let (bp_a, canon_a) = i256_pair(&a, dtype);
        let (bp_b, canon_b) = i256_pair(&b, dtype);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let native = bp_a
            .binary(bp_b, operator)?
            .execute::<BoolArray>(&mut ctx)?;
        let canonical = canon_a
            .binary(canon_b, operator)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(native.into_array(), canonical.into_array());
        Ok(())
    }
}
