// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use super::SumState;
use crate::arrays::DecimalArray;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;

/// Accumulate a decimal array into the sum state.
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
pub(super) fn accumulate_decimal(inner: &mut SumState, d: &DecimalArray) -> VortexResult<bool> {
    let SumState::Decimal(acc) = inner else {
        vortex_panic!("expected decimal sum state for decimal input");
    };

    let mask = d.validity_mask()?;
    match mask.bit_buffer() {
        AllOr::None => Ok(false),
        AllOr::All => match_each_decimal_value_type!(d.values_type(), |T| {
            for &v in d.buffer::<T>().iter() {
                match acc.checked_add(&DecimalValue::from(v)) {
                    Some(r) => *acc = r,
                    None => return Ok(true),
                }
            }
            Ok(false)
        }),
        AllOr::Some(validity) => match_each_decimal_value_type!(d.values_type(), |T| {
            for (&v, valid) in d.buffer::<T>().iter().zip_eq(validity.iter()) {
                if valid {
                    match acc.checked_add(&DecimalValue::from(v)) {
                        Some(r) => *acc = r,
                        None => return Ok(true),
                    }
                }
            }
            Ok(false)
        }),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
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
}
