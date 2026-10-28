// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

/// Compute `scalar * len` for a constant array, returning the product as a sum-typed scalar.
///
/// Returns `Ok(None)` if the scalar is null (no contribution to the sum).
/// Returns a null scalar on overflow (saturation).
pub(crate) fn multiply_constant(
    scalar: &Scalar,
    len: usize,
    return_dtype: &DType,
) -> VortexResult<Option<Scalar>> {
    if scalar.is_null() || len == 0 {
        return Ok(None);
    }

    let product = match scalar.dtype() {
        DType::Bool(_) => {
            let val = scalar
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("Expected non-null bool scalar for sum"))?;
            if !val {
                return Ok(None);
            }
            Scalar::primitive(len as u64, Nullability::Nullable)
        }
        DType::Primitive(..) => {
            let pvalue = scalar
                .as_primitive()
                .pvalue()
                .ok_or_else(|| vortex_err!("Expected non-null primitive scalar for sum"))?;
            match return_dtype {
                DType::Primitive(PType::U64, _) => {
                    let val = pvalue.cast::<u64>()?;
                    match val.checked_mul(len as u64) {
                        Some(product) => Scalar::primitive(product, Nullability::Nullable),
                        None => Scalar::null(return_dtype.as_nullable()),
                    }
                }
                DType::Primitive(PType::I64, _) => {
                    let val = pvalue.cast::<i64>()?;
                    match i64::try_from(len).ok().and_then(|l| val.checked_mul(l)) {
                        Some(product) => Scalar::primitive(product, Nullability::Nullable),
                        None => Scalar::null(return_dtype.as_nullable()),
                    }
                }
                DType::Primitive(PType::F64, _) => {
                    let val = pvalue.cast::<f64>()?;
                    Scalar::primitive(val * len as f64, Nullability::Nullable)
                }
                _ => vortex_bail!(
                    "Unexpected return dtype for primitive sum: {}",
                    return_dtype
                ),
            }
        }
        DType::Decimal(..) => {
            let val = scalar
                .as_decimal()
                .decimal_value()
                .ok_or_else(|| vortex_err!("Expected non-null decimal scalar for sum"))?;
            let ret_decimal = *return_dtype
                .as_decimal_opt()
                .ok_or_else(|| vortex_err!("Expected decimal return dtype"))?;

            // Multiply in i256, the widest storage type, because the operands can be narrower
            // than the return type. Then check the product against the return precision.
            let val = DecimalValue::I256(val.as_i256());
            let len_decimal = DecimalValue::I256(i256::from_i128(len as i128));
            match val
                .checked_mul(&len_decimal)
                .and_then(|product| product.normalize(ret_decimal))
            {
                Some(product) => Scalar::decimal(product, ret_decimal, Nullability::Nullable),
                None => Scalar::null(return_dtype.as_nullable()),
            }
        }
        _ => vortex_bail!("Unsupported constant type for sum: {}", scalar.dtype()),
    };

    Ok(Some(product))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
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

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn sum_constant_unsigned() -> VortexResult<()> {
        let array = ConstantArray::new(5u64, 10).into_array();
        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result, 50u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_signed() -> VortexResult<()> {
        let array = ConstantArray::new(-5i64, 10).into_array();
        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result, (-50i64).into());
        Ok(())
    }

    #[test]
    fn sum_constant_nullable_value() -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10)
            .into_array();
        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_constant_bool_false() -> VortexResult<()> {
        let array = ConstantArray::new(false, 10).into_array();
        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result, 0u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_bool_true() -> VortexResult<()> {
        let array = ConstantArray::new(true, 10).into_array();
        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result, 10u64.into());
        Ok(())
    }

    #[test]
    fn sum_constant_bool_null() -> VortexResult<()> {
        let array = ConstantArray::new(Scalar::null(DType::Bool(Nullable)), 10).into_array();
        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
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

        let result = sum(&array, &mut SESSION.create_execution_ctx())?;

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

        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
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

        let result = sum(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_900)))
        );
        Ok(())
    }

    #[test]
    fn sum_constant_decimal_i128_value_overflows_i128() -> VortexResult<()> {
        let value: i128 = 112_285_607_173_948_541_047_296_740_235_646_205_996;
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(value),
                DecimalDType::new(39, 39),
                Nullability::NonNullable,
            ),
            2,
        )
        .into_array();

        let result = sum(&array, &mut SESSION.create_execution_ctx())?;

        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(
                i256::from_i128(value) + i256::from_i128(value)
            ))
        );
        Ok(())
    }

    #[test]
    fn sum_constant_decimal_exceeds_return_precision() -> VortexResult<()> {
        let max_value = i256::from_i128(10).checked_pow(76).unwrap() - i256::from_i128(1);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I256(max_value),
                DecimalDType::new(76, 0),
                Nullability::NonNullable,
            ),
            2,
        )
        .into_array();

        let result = sum(&array, &mut SESSION.create_execution_ctx())?;

        assert!(result.is_null());
        Ok(())
    }
}
