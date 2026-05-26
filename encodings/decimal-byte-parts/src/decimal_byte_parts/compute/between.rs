// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::compute::compare::decimal_value_wrapper_to_primitive;

impl BetweenKernel for DecimalByteParts {
    fn between(
        arr: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
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
                ctx,
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

/// Evaluate `lower <= value <= upper` (respecting strictness) over the two-limb representation in a
/// single fused pass.
///
/// With high limb `H` (signed i64) and low limb `L` (unsigned u64), the value `v = H<<64 | L`
/// satisfies `v >= lower` iff `H > lo_h OR (H == lo_h AND L >=' lo_l)` and `v <= upper` iff
/// `H < hi_h OR (H == hi_h AND L <=' hi_l)`, where `>='`/`<='` are strict when requested. Each
/// row resolves to native-width integer comparisons in one loop, which vectorizes far better than
/// arrow's 128-bit comparison or a tree of generic array operations with intermediate allocations.
fn two_limb_between(
    arr: &ArrayView<'_, DecimalByteParts>,
    lower: i128,
    upper: i128,
    options: &BetweenOptions,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let high = arr.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let low = arr
        .lower()
        .vortex_expect("two-limb path requires a lower limb")
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    let validity = high.validity()?.union_nullability(nullability);
    let high = high.as_slice::<i64>();
    let low = low.as_slice::<u64>();
    assert_eq!(high.len(), low.len(), "limb lengths must match");

    let (lower_high, lower_low) = split_i128(lower);
    let (upper_high, upper_low) = split_i128(upper);

    let lower_limbs = (lower_high, lower_low);
    let upper_limbs = (upper_high, upper_low);

    // Pass the low-limb comparison as a monomorphized fn so the whole per-element body inlines.
    let bits = match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => {
            collect_two_limb(high, low, lower_limbs, u64_gt, upper_limbs, u64_lt)
        }
        (StrictComparison::Strict, StrictComparison::NonStrict) => {
            collect_two_limb(high, low, lower_limbs, u64_gt, upper_limbs, u64_le)
        }
        (StrictComparison::NonStrict, StrictComparison::Strict) => {
            collect_two_limb(high, low, lower_limbs, u64_ge, upper_limbs, u64_lt)
        }
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => {
            collect_two_limb(high, low, lower_limbs, u64_ge, upper_limbs, u64_le)
        }
    };

    Ok(BoolArray::new(bits, validity).into_array())
}

/// The fused per-element loop. `lower_low_cmp`/`upper_low_cmp` apply the (possibly strict) low-limb
/// comparison. Bitwise (non-short-circuiting) `&`/`|` keep the body branch-free for the vectorizer.
fn collect_two_limb(
    high: &[i64],
    low: &[u64],
    lower: (i64, u64),
    lower_low_cmp: impl Fn(u64, u64) -> bool,
    upper: (i64, u64),
    upper_low_cmp: impl Fn(u64, u64) -> bool,
) -> BitBuffer {
    let (lower_high, lower_low) = lower;
    let (upper_high, upper_low) = upper;
    BitBuffer::collect_bool(high.len(), |idx| {
        // SAFETY: collect_bool yields idx in 0..high.len(), and low.len() == high.len().
        let h = unsafe { *high.get_unchecked(idx) };
        let l = unsafe { *low.get_unchecked(idx) };
        let ge_lower = (h > lower_high) | ((h == lower_high) & lower_low_cmp(l, lower_low));
        let le_upper = (h < upper_high) | ((h == upper_high) & upper_low_cmp(l, upper_low));
        ge_lower & le_upper
    })
}

fn u64_ge(a: u64, b: u64) -> bool {
    a >= b
}
fn u64_gt(a: u64, b: u64) -> bool {
    a > b
}
fn u64_le(a: u64, b: u64) -> bool {
    a <= b
}
fn u64_lt(a: u64, b: u64) -> bool {
    a < b
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
