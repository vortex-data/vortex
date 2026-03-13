// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use super::SumState;
use super::checked_add_i64;
use super::checked_add_u64;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar::DecimalValue;

/// Accumulate a constant array into the sum state.
/// Computes `scalar * len` and adds to the accumulator.
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
pub(super) fn accumulate_constant(inner: &mut SumState, c: &ConstantArray) -> VortexResult<bool> {
    let scalar = c.scalar();
    if scalar.is_null() || c.is_empty() {
        return Ok(false);
    }
    let len = c.len();

    match scalar.dtype() {
        DType::Bool(_) => {
            let SumState::Unsigned(acc) = inner else {
                vortex_panic!("expected unsigned sum state for bool input");
            };
            let val = scalar
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("Expected non-null bool scalar for sum"))?;
            if val {
                Ok(checked_add_u64(acc, len as u64))
            } else {
                Ok(false)
            }
        }
        DType::Primitive(..) => {
            let pvalue = scalar
                .as_primitive()
                .pvalue()
                .ok_or_else(|| vortex_err!("Expected non-null primitive scalar for sum"))?;
            match inner {
                SumState::Unsigned(acc) => {
                    let val = pvalue.cast::<u64>()?;
                    match val.checked_mul(len as u64) {
                        Some(product) => Ok(checked_add_u64(acc, product)),
                        None => Ok(true),
                    }
                }
                SumState::Signed(acc) => {
                    let val = pvalue.cast::<i64>()?;
                    match i64::try_from(len).ok().and_then(|l| val.checked_mul(l)) {
                        Some(product) => Ok(checked_add_i64(acc, product)),
                        None => Ok(true),
                    }
                }
                SumState::Float(acc) => {
                    let val = pvalue.cast::<f64>()?;
                    *acc += val * len as f64;
                    Ok(false)
                }
                SumState::Decimal(_) => {
                    vortex_panic!("decimal sum state with primitive input")
                }
            }
        }
        DType::Decimal(..) => {
            let SumState::Decimal(acc) = inner else {
                vortex_panic!("expected decimal sum state for decimal input");
            };
            let val = scalar
                .as_decimal()
                .decimal_value()
                .ok_or_else(|| vortex_err!("Expected non-null decimal scalar for sum"))?;
            let len_decimal = DecimalValue::from(len as i128);
            match val.checked_mul(&len_decimal) {
                Some(product) => match acc.checked_add(&product) {
                    Some(r) => {
                        *acc = r;
                        Ok(false)
                    }
                    None => Ok(true),
                },
                None => Ok(true),
            }
        }
        _ => vortex_bail!("Unsupported constant type for sum: {}", scalar.dtype()),
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::arrays::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::expr::stats::Stat;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;

    #[test]
    fn sum_constant_unsigned() -> VortexResult<()> {
        let array = ConstantArray::new(5u64, 10).into_array();
        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, 50u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_signed() -> VortexResult<()> {
        let array = ConstantArray::new(-5i64, 10).into_array();
        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, (-50i64).into());
        Ok(())
    }

    #[test]
    fn sum_constant_nullable_value() -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10)
            .into_array();
        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_constant_bool_false() -> VortexResult<()> {
        let array = ConstantArray::new(false, 10).into_array();
        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, 0u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_bool_true() -> VortexResult<()> {
        let array = ConstantArray::new(true, 10).into_array();
        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, 10u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_bool_null() -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::null(DType::Bool(Nullable)), 10).into_array();
        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_constant_decimal() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(100),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            5,
        )
        .into_array();

        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;

        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(500)))
        );
        assert_eq!(result.dtype(), &Stat::Sum.dtype(array.dtype()).unwrap());
        Ok(())
    }

    #[test]
    fn sum_constant_decimal_null() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(Scalar::null(DType::Decimal(decimal_dtype, Nullable)), 10)
            .into_array();

        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(
            result,
            Scalar::decimal(
                DecimalValue::I256(i256::ZERO),
                DecimalDType::new(20, 2),
                Nullable
            )
        );
        Ok(())
    }

    #[test]
    fn sum_constant_decimal_large_value() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(999_999_999),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            100,
        )
        .into_array();

        let result = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_900)))
        );
        Ok(())
    }
}
