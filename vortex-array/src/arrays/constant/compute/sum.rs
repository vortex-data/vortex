// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrowNativeTypeOp;
use num_traits::{CheckedAdd, CheckedMul, ToPrimitive};
use vortex_dtype::{DType, DecimalDType, NativePType, Nullability, i256, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{DecimalScalar, DecimalValue, PrimitiveScalar, Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;
use crate::stats::Stat;

impl SumKernel for ConstantVTable {
    fn sum(&self, array: &ConstantArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        // Compute the expected dtype of the sum.
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;

        let sum_value = sum_scalar(array.scalar(), array.len(), accumulator)?;
        Ok(Scalar::new(sum_dtype, sum_value))
    }
}

fn sum_scalar(scalar: &Scalar, len: usize, accumulator: &Scalar) -> VortexResult<ScalarValue> {
    match scalar.dtype() {
        DType::Bool(_) => {
            let count = match scalar.as_bool().value() {
                None => unreachable!("Handled before reaching this point"),
                Some(false) => 0u64,
                Some(true) => len as u64,
            };
            let accumulator = accumulator
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("cannot be null");
            Ok(ScalarValue::from(accumulator.checked_add(count)))
        }
        DType::Primitive(ptype, _) => {
            let result = match_each_native_ptype!(
                ptype,
                unsigned: |T| { sum_integral::<u64>(scalar.as_primitive(), len, accumulator)?.into() },
                signed: |T| { sum_integral::<i64>(scalar.as_primitive(), len, accumulator)?.into() },
                floating: |T| { sum_float(scalar.as_primitive(), len, accumulator)?.into() }
            );
            Ok(result)
        }
        DType::Decimal(decimal_dtype, _) => {
            sum_decimal(scalar.as_decimal(), len, *decimal_dtype, accumulator)
        }
        DType::Extension(_) => sum_scalar(&scalar.as_extension().storage(), len, accumulator),
        dtype => vortex_bail!("Unsupported dtype for sum: {}", dtype),
    }
}

fn sum_decimal(
    decimal_scalar: DecimalScalar,
    array_len: usize,
    decimal_dtype: DecimalDType,
    accumulator: &Scalar,
) -> VortexResult<ScalarValue> {
    let result_dtype = Stat::Sum
        .dtype(&DType::Decimal(decimal_dtype, Nullability::Nullable))
        .vortex_expect("decimal supports sum");
    let result_decimal_type = result_dtype
        .as_decimal_opt()
        .vortex_expect("must be decimal");

    let Some(value) = decimal_scalar.decimal_value() else {
        // Null value: return null
        return Ok(ScalarValue::null());
    };

    // Convert array_len to DecimalValue for multiplication
    let len_value = DecimalValue::I256(i256::from_i128(array_len as i128));

    // Multiply value * len
    let array_sum = value.checked_mul(&len_value).and_then(|result| {
        // Check if result fits in the precision
        result
            .fits_in_precision(*result_decimal_type)
            .unwrap_or(false)
            .then_some(result)
    });

    // Add accumulator to array_sum
    let initial_decimal = DecimalScalar::try_from(accumulator)?;
    let initial_dec_value = initial_decimal
        .decimal_value()
        .unwrap_or(DecimalValue::I256(i256::ZERO));

    match array_sum {
        Some(array_sum_value) => {
            let total = array_sum_value
                .checked_add(&initial_dec_value)
                .and_then(|result| {
                    result
                        .fits_in_precision(*result_decimal_type)
                        .unwrap_or(false)
                        .then_some(result)
                });
            match total {
                Some(result_value) => Ok(ScalarValue::from(result_value)),
                None => Ok(ScalarValue::null()), // Overflow
            }
        }
        None => Ok(ScalarValue::null()), // Overflow
    }
}

fn sum_integral<T>(
    primitive_scalar: PrimitiveScalar<'_>,
    array_len: usize,
    accumulator: &Scalar,
) -> VortexResult<Option<T>>
where
    T: NativePType + CheckedMul + CheckedAdd,
    Scalar: From<Option<T>>,
{
    let v = primitive_scalar.as_::<T>();
    let array_len =
        T::from(array_len).ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;
    let Some(array_sum) = v.and_then(|v| v.checked_mul(&array_len)) else {
        return Ok(None);
    };

    let initial = accumulator
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");
    Ok(initial.checked_add(&array_sum))
}

fn sum_float(
    primitive_scalar: PrimitiveScalar<'_>,
    array_len: usize,
    accumulator: &Scalar,
) -> VortexResult<Option<f64>> {
    let v = primitive_scalar
        .as_::<f64>()
        .vortex_expect("cannot be null");
    let array_len = array_len
        .to_f64()
        .ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;

    let Ok(array_sum) = v.mul_checked(array_len) else {
        return Ok(None);
    };
    let initial = accumulator
        .as_primitive()
        .as_::<f64>()
        .vortex_expect("cannot be null");
    Ok(Some(initial + array_sum))
}

register_kernel!(SumKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, DecimalDType, Nullability, PType, i256};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::arrays::ConstantArray;
    use crate::compute::sum;
    use crate::stats::Stat;
    use crate::{Array, IntoArray};

    #[test]
    fn test_sum_unsigned() {
        let array = ConstantArray::new(5u64, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, 50u64.into());
    }

    #[test]
    fn test_sum_signed() {
        let array = ConstantArray::new(-5i64, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, (-50i64).into());
    }

    #[test]
    fn test_sum_nullable_value() {
        let array = ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10)
            .into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
    }

    #[test]
    fn test_sum_bool_false() {
        let array = ConstantArray::new(false, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, 0u64.into());
    }

    #[test]
    fn test_sum_bool_true() {
        let array = ConstantArray::new(true, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, 10u64.into());
    }

    #[test]
    fn test_sum_bool_null() {
        let array = ConstantArray::new(Scalar::null(DType::Bool(Nullable)), 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
    }

    #[test]
    fn test_sum_decimal() {
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

        let result = sum(&array).unwrap();

        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(500)))
        );
        assert_eq!(result.dtype(), &Stat::Sum.dtype(array.dtype()).unwrap());
    }

    #[test]
    fn test_sum_decimal_null() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(Scalar::null(DType::Decimal(decimal_dtype, Nullable)), 10)
            .into_array();

        let result = sum(&array).unwrap();
        assert_eq!(
            result,
            Scalar::decimal(
                DecimalValue::I256(i256::ZERO),
                DecimalDType::new(20, 2),
                Nullable
            )
        );
    }

    #[test]
    fn test_sum_decimal_large_value() {
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

        let result = sum(&array).unwrap();
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_900)))
        );
    }
}
