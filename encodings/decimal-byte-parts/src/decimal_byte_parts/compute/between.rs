// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::LOWER_PTYPE;
use crate::decimal_byte_parts::compute::compare::decimal_value_wrapper_to_primitive;

impl BetweenKernel for DecimalByteParts {
    fn between(
        arr: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // We can only push the comparison down into the limbs when both bounds are constant.
        let (Some(lower_const), Some(upper_const)) = (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };

        // NOTE: the `between` entrypoint precondition already replaced null bounds with an
        // all-null result, so both bounds are guaranteed to be non-null here.
        let lower_decimal = lower_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");
        let upper_decimal = upper_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        if arr.lower().is_some() {
            // Two-limb representation: a lexicographic comparison over the (signed high, unsigned
            // low) limbs. Both bounds must fit in i128 to be split into limbs.
            let (Some(lower_i128), Some(upper_i128)) =
                (lower_decimal.cast::<i128>(), upper_decimal.cast::<i128>())
            else {
                return Ok(None);
            };
            return Ok(Some(two_limb_between(
                &arr,
                lower_i128,
                upper_i128,
                options,
                nullability,
            )?));
        }

        let scalar_type = arr.msp().dtype().with_nullability(nullability);
        let msp_ptype = arr.msp().as_primitive_typed().ptype();

        // If either bound falls outside the MSP's physical integer range we cannot push the
        // comparison down losslessly. Fall back to the canonical decimal `between`, which handles
        // the overflow directions (all-true / all-false constraints) correctly.
        let (Ok(lower_value), Ok(upper_value)) = (
            decimal_value_wrapper_to_primitive(lower_decimal, msp_ptype),
            decimal_value_wrapper_to_primitive(upper_decimal, msp_ptype),
        ) else {
            return Ok(None);
        };

        let lower_const = ConstantArray::new(
            Scalar::try_new(scalar_type.clone(), Some(lower_value))?,
            arr.len(),
        );
        let upper_const =
            ConstantArray::new(Scalar::try_new(scalar_type, Some(upper_value))?, arr.len());

        arr.msp()
            .clone()
            .between(
                lower_const.into_array(),
                upper_const.into_array(),
                options.clone(),
            )
            .map(Some)
    }
}

/// Evaluate `lower <= value <= upper` (respecting strictness) over the two-limb representation.
///
/// With high limb `H` (signed i64) and low limb `L` (unsigned u64), the value `v = H<<64 | L`
/// satisfies `v >= lower` iff `H > lo_h OR (H == lo_h AND L >=' lo_l)` and `v <= upper` iff
/// `H < hi_h OR (H == hi_h AND L <=' hi_l)`, where `>='`/`<='` are strict when requested. Each
/// limb comparison is a native-width integer compare that vectorizes far better than a 128-bit
/// comparison.
fn two_limb_between(
    arr: &ArrayView<'_, DecimalByteParts>,
    lower: i128,
    upper: i128,
    options: &BetweenOptions,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = arr.len();
    let high = arr.msp().clone();
    let low = arr
        .lower()
        .vortex_expect("two-limb path requires a lower limb")
        .clone();

    let high_dtype = high.dtype().with_nullability(nullability);
    let low_dtype = DType::Primitive(LOWER_PTYPE, nullability);

    let high_const = |limb: i64| -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(
            Scalar::try_new(high_dtype.clone(), Some(ScalarValue::from(limb)))?,
            len,
        )
        .into_array())
    };
    let low_const = |limb: u64| -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(
            Scalar::try_new(low_dtype.clone(), Some(ScalarValue::from(limb)))?,
            len,
        )
        .into_array())
    };

    let lower_low_op = match options.lower_strict {
        StrictComparison::Strict => Operator::Gt,
        StrictComparison::NonStrict => Operator::Gte,
    };
    let upper_low_op = match options.upper_strict {
        StrictComparison::Strict => Operator::Lt,
        StrictComparison::NonStrict => Operator::Lte,
    };

    let (lower_high, lower_low) = split_i128(lower);
    let (upper_high, upper_low) = split_i128(upper);

    // value >= lower
    let ge_lower = {
        let h_gt = high.binary(high_const(lower_high)?, Operator::Gt)?;
        let h_eq = high.binary(high_const(lower_high)?, Operator::Eq)?;
        let l_cmp = low.binary(low_const(lower_low)?, lower_low_op)?;
        h_gt.binary(h_eq.binary(l_cmp, Operator::And)?, Operator::Or)?
    };

    // value <= upper
    let le_upper = {
        let h_lt = high.binary(high_const(upper_high)?, Operator::Lt)?;
        let h_eq = high.binary(high_const(upper_high)?, Operator::Eq)?;
        let l_cmp = low.binary(low_const(upper_low)?, upper_low_op)?;
        h_lt.binary(h_eq.binary(l_cmp, Operator::And)?, Operator::Or)?
    };

    ge_lower.binary(le_upper, Operator::And)
}

/// Split an i128 into its signed high and unsigned low 64-bit limbs. The truncating casts are the
/// intended limb extraction: `>> 64` keeps the high 64 bits and `as u64` keeps the low 64 bits.
#[allow(clippy::cast_possible_truncation)]
fn split_i128(value: i128) -> (i64, u64) {
    ((value >> 64) as i64, value as u64)
}

#[cfg(test)]
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
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::DecimalByteParts;

    fn decimal_const(value: DecimalValue, decimal_type: DecimalDType, len: usize) -> ArrayRef {
        ConstantArray::new(
            Scalar::decimal(value, decimal_type, Nullability::NonNullable),
            len,
        )
        .into_array()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn two_limb(values: &[i128], decimal_type: DecimalDType) -> ArrayRef {
        let highs: Buffer<i64> = values.iter().map(|v| (v >> 64) as i64).collect();
        let lows: Buffer<u64> = values.iter().map(|v| *v as u64).collect();
        DecimalByteParts::try_new_with_lower(
            PrimitiveArray::new(highs, Validity::NonNullable).into_array(),
            PrimitiveArray::new(lows, Validity::NonNullable).into_array(),
            decimal_type,
        )
        .unwrap()
        .into_array()
    }

    /// The two-limb `between` pushdown must agree with the canonical i128 implementation across
    /// values spanning the low-limb wraparound, the high limb, and negatives, for every strictness.
    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict)]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict)]
    #[case(StrictComparison::Strict, StrictComparison::Strict)]
    fn two_limb_between_matches_canonical(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decimal_type = DecimalDType::new(38, 0);
        let values: Vec<i128> = vec![
            0,
            1,
            -1,
            i128::from(i64::MAX),
            i128::from(i64::MAX) + 1,
            (5i128 << 64) | 3,
            (5i128 << 64) | 5,
            (5i128 << 64) | 9,
            (4i128 << 64) | i128::from(u64::MAX),
            (6i128 << 64),
            -(7i128 << 64) | 11,
        ];
        let lower = (5i128 << 64) | 3;
        let upper = (5i128 << 64) | 9;
        let len = values.len();
        let options = BetweenOptions {
            lower_strict,
            upper_strict,
        };

        let lower_arr = decimal_const(DecimalValue::I128(lower), decimal_type, len);
        let upper_arr = decimal_const(DecimalValue::I128(upper), decimal_type, len);

        let got = two_limb(&values, decimal_type)
            .between(lower_arr.clone(), upper_arr.clone(), options.clone())?
            .execute::<BoolArray>(&mut ctx)?;

        let canonical = DecimalArray::new(
            values.iter().copied().collect::<Buffer<i128>>(),
            decimal_type,
            Validity::NonNullable,
        )
        .into_array();
        let want = canonical
            .between(lower_arr, upper_arr, options)?
            .execute::<BoolArray>(&mut ctx)?;

        assert_arrays_eq!(got, want);
        Ok(())
    }

    /// A two-limb array must canonicalize to the same values as a canonical i128 `DecimalArray`.
    #[test]
    fn two_limb_canonicalizes_to_i128() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decimal_type = DecimalDType::new(38, 0);
        let values: Vec<i128> = vec![
            0,
            -1,
            i128::from(i64::MIN),
            (3i128 << 64) | 42,
            -(9i128 << 64) | 17,
        ];

        let got = two_limb(&values, decimal_type).execute::<DecimalArray>(&mut ctx)?;
        let want = DecimalArray::new(
            values.iter().copied().collect::<Buffer<i128>>(),
            decimal_type,
            Validity::NonNullable,
        );
        assert_arrays_eq!(got.into_array(), want.into_array());
        Ok(())
    }

    #[test]
    fn between_decimal_const() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(8, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300, 400, 500], Validity::AllValid)
                .into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(200), decimal_type, arr.len());
        let upper = decimal_const(DecimalValue::I64(400), decimal_type, arr.len());

        // 200 <= value <= 400
        let res = arr.clone().between(
            lower.clone(),
            upper.clone(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(true), Some(true), Some(false)])
        );

        // 200 < value < 400
        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::Strict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false)
            ])
        );

        Ok(())
    }

    #[test]
    fn between_decimal_nullable() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(8, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200, 300, 400],
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            )
            .into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(100), decimal_type, arr.len());
        let upper = decimal_const(DecimalValue::I64(300), decimal_type, arr.len());

        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([None, Some(true), Some(true), Some(false)])
        );

        Ok(())
    }

    /// Bounds that do not fit in the MSP's physical type must fall back to the canonical decimal
    /// `between`, which handles the overflow directions. Here the array uses i32 storage but the
    /// upper bound only fits in i128, so the upper constraint is always satisfied.
    #[test]
    fn between_decimal_unconvertible_bound() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(38, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300], Validity::AllValid).into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(150), decimal_type, arr.len());
        let upper = decimal_const(
            DecimalValue::I128(9_999_999_999_999_999_999),
            decimal_type,
            arr.len(),
        );

        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(true)])
        );

        Ok(())
    }
}
