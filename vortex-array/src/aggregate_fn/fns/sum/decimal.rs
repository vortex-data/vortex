// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::CheckedAdd;
use num_traits::NumOps;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::SumState;
use crate::arrays::DecimalArray;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;

/// Accumulate a decimal array into the sum state.
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
pub(super) fn accumulate_decimal(inner: &mut SumState, d: &DecimalArray) -> VortexResult<bool> {
    let mask = d.validity_mask()?;
    let validity = match &mask {
        Mask::AllTrue(_) => None,
        Mask::Values(mask_values) => Some(mask_values.bit_buffer()),
        Mask::AllFalse(_) => {
            return Ok(false);
        }
    };

    let SumState::Decimal { value, dtype } = inner else {
        vortex_panic!("expected decimal sum state for decimal input");
    };

    let values_type = DecimalType::smallest_decimal_value_type(dtype);
    match_each_decimal_value_type!(d.values_type(), |T| {
        match_each_decimal_value_type!(values_type, |I| {
            let initial: I = value
                .cast()
                .vortex_expect("cannot fail to cast initial value");
            match sum_decimal_value(initial, d.buffer::<T>(), validity, *dtype) {
                Some(v) => *value = v,
                None => return Ok(true),
            }
            Ok(false)
        })
    })
}

fn sum_decimal_value<T, I>(
    initial: I,
    values: Buffer<T>,
    validity: Option<&BitBuffer>,
    output_dtype: DecimalDType,
) -> Option<DecimalValue>
where
    T: AsPrimitive<I>,
    I: NumOps + CheckedAdd + Copy + NativeDecimalType + 'static,
    bool: AsPrimitive<I>,
    DecimalValue: From<I>,
{
    let sum = match validity {
        Some(v) => sum_decimal_with_validity(values, v, initial),
        None => sum_decimal(values, initial),
    };

    sum.map(DecimalValue::from)
        // We have to make sure that the decimal value fits the precision of the decimal dtype.
        .filter(|v| v.fits_in_precision(output_dtype))
}

fn sum_decimal<T: AsPrimitive<I>, I: Copy + CheckedAdd + 'static>(
    values: Buffer<T>,
    initial: I,
) -> Option<I> {
    let mut sum = initial;
    for v in values.iter() {
        let v: I = v.as_();
        sum = CheckedAdd::checked_add(&sum, &v)?;
    }
    Some(sum)
}

fn sum_decimal_with_validity<T, I>(values: Buffer<T>, validity: &BitBuffer, initial: I) -> Option<I>
where
    T: AsPrimitive<I>,
    I: NumOps + CheckedAdd + Copy + 'static,
    bool: AsPrimitive<I>,
{
    let mut sum = initial;
    for (v, valid) in values.iter().zip_eq(validity) {
        let v: I = v.as_() * valid.as_();

        sum = CheckedAdd::checked_add(&sum, &v)?;
    }
    Some(sum)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::arrays::DecimalArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn sum_decimal_basic() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(600i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_with_nulls() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true, true]),
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullable),
            Some(ScalarValue::from(DecimalValue::from(800i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_negative_values() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, -200i32, 300i32, -50i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(150i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_near_i32_max() -> VortexResult<()> {
        let near_max = i32::MAX - 1000;
        let decimal = DecimalArray::new(
            buffer![near_max, 500i32, 400i32],
            DecimalDType::new(10, 2),
            Validity::AllValid,
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected_sum = near_max as i64 + 500 + 400;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(20, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_large_i64_values() -> VortexResult<()> {
        let large_val = i64::MAX / 4;
        let decimal = DecimalArray::new(
            buffer![large_val, large_val, large_val, large_val + 1],
            DecimalDType::new(19, 0),
            Validity::AllValid,
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected_sum = (large_val as i128) * 4 + 1;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(29, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_preserves_scale() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![12345i32, 67890i32, 11111i32],
            DecimalDType::new(6, 4),
            Validity::AllValid,
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(16, 4), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(91346i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_single_value() -> VortexResult<()> {
        let decimal =
            DecimalArray::new(buffer![42i32], DecimalDType::new(3, 1), Validity::AllValid);

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(13, 1), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(42i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_all_nulls_except_one() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([false, false, true, false]),
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullable),
            Some(ScalarValue::from(DecimalValue::from(300i32))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_overflow_detection() -> VortexResult<()> {
        let max_val = i128::MAX / 2;
        let decimal = DecimalArray::new(
            buffer![max_val, max_val, max_val],
            DecimalDType::new(38, 0),
            Validity::AllValid,
        );

        let result = sum(
            &decimal.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        let expected_sum =
            i256::from_i128(max_val) + i256::from_i128(max_val) + i256::from_i128(max_val);
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(48, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn sum_decimal_i256_overflow() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(76, 0);
        let decimal = DecimalArray::new(
            buffer![i256::MAX, i256::MAX, i256::MAX],
            decimal_dtype,
            Validity::AllValid,
        );

        assert_eq!(
            sum(
                &decimal.into_array(),
                &mut LEGACY_SESSION.create_execution_ctx()
            )
            .vortex_expect("operation should succeed in test"),
            Scalar::null(DType::Decimal(decimal_dtype, Nullable))
        );
        Ok(())
    }

    #[test]
    fn sum_decimal_near_precision_boundary() -> VortexResult<()> {
        // Input precision 4 → return precision min(76, 4+10) = 14.
        // Native type for precision 14 is I64 (max precision 18), so 14 < 18.
        // Use combine_partials to push state near (but under) 10^14.
        let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let mut state = Sum.empty_partial(&EmptyOptions, &input_dtype)?;

        let near_limit = Scalar::decimal(
            DecimalValue::from(99_999_999_999_990i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        Sum.combine_partials(&mut state, near_limit)?;

        // Add a small value that keeps us just under 10^14.
        let small = Scalar::decimal(DecimalValue::from(9i64), DecimalDType::new(14, 0), Nullable);
        Sum.combine_partials(&mut state, small)?;

        let result = Sum.to_scalar(&state)?;
        assert!(!result.is_null());
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_999_999)))
        );
        Ok(())
    }

    #[test]
    fn sum_decimal_precision_overflow_within_i256() -> VortexResult<()> {
        // Input precision 4 → return precision 14. Native I64 (max 18).
        // The max representable value for precision 14 is 10^14 - 1.
        // When the sum reaches exactly 10^14, fits_in_precision fails even though
        // i256 arithmetic does not overflow. This tests the precision-based
        // saturation path in combine_partials.
        let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let mut state = Sum.empty_partial(&EmptyOptions, &input_dtype)?;

        let near_limit = Scalar::decimal(
            DecimalValue::from(99_999_999_999_999i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        Sum.combine_partials(&mut state, near_limit)?;

        // Push the sum to exactly 10^14, exceeding precision 14.
        let one_more =
            Scalar::decimal(DecimalValue::from(1i64), DecimalDType::new(14, 0), Nullable);
        Sum.combine_partials(&mut state, one_more)?;

        let result = Sum.to_scalar(&state)?;
        assert!(result.is_null());
        assert_eq!(
            result.dtype(),
            &DType::Decimal(DecimalDType::new(14, 0), Nullable)
        );
        Ok(())
    }

    #[test]
    fn sum_decimal_precision_overflow_negative() -> VortexResult<()> {
        // Same setup but with negative values: sum reaches -10^14.
        let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
        let mut state = Sum.empty_partial(&EmptyOptions, &input_dtype)?;

        let near_limit = Scalar::decimal(
            DecimalValue::from(-99_999_999_999_999i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        Sum.combine_partials(&mut state, near_limit)?;

        let one_more = Scalar::decimal(
            DecimalValue::from(-1i64),
            DecimalDType::new(14, 0),
            Nullable,
        );
        Sum.combine_partials(&mut state, one_more)?;

        let result = Sum.to_scalar(&state)?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_decimal_accumulate_precision_overflow() -> VortexResult<()> {
        // Test precision overflow via the accumulate_decimal path (not combine_partials).
        // Input precision 28 (I128 storage) → return precision min(76, 38) = 38.
        // Native for precision 38 is I128 (max 38), so 38 = 38.
        // Use precision 27 → return 37. Native for 37 is I128 (max 38), so 37 < 38.
        //
        // We use combine_partials to get the state close to 10^37, then accumulate
        // a real array that pushes it over.
        let input_dtype = DType::Decimal(DecimalDType::new(27, 0), Nullability::NonNullable);
        let return_dtype = DecimalDType::new(37, 0);
        let mut state = Sum.empty_partial(&EmptyOptions, &input_dtype)?;

        // Set state to 10^37 - 1 via combine_partials.
        let near_limit_val: i128 = 10i128.pow(37) - 1;
        let near_limit =
            Scalar::decimal(DecimalValue::from(near_limit_val), return_dtype, Nullable);
        Sum.combine_partials(&mut state, near_limit)?;

        // Now accumulate a real i128 array with a single element = 1 to overflow precision.
        let decimal =
            DecimalArray::new(buffer![1i128], DecimalDType::new(27, 0), Validity::AllValid);

        // Drive accumulate through the vtable directly.
        let columnar = crate::Columnar::Canonical(crate::Canonical::Decimal(decimal));
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        Sum.accumulate(&mut state, &columnar, &mut ctx)?;

        let result = Sum.to_scalar(&state)?;
        assert!(result.is_null());
        Ok(())
    }
}
