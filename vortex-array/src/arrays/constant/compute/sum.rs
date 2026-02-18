// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::CheckedAdd;
use num_traits::CheckedMul;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::expr::stats::Stat;
use crate::match_each_native_ptype;
use crate::register_kernel;
use crate::scalar::DecimalScalar;
use crate::scalar::DecimalValue;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl SumKernel for ConstantVTable {
    fn sum(&self, array: &ConstantArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        // Compute the expected dtype of the sum.
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;

        let sum_value = sum_scalar(array.scalar(), array.len(), accumulator)?;
        Scalar::try_new(sum_dtype, sum_value)
    }
}

fn sum_scalar(
    scalar: &Scalar,
    len: usize,
    accumulator: &Scalar,
) -> VortexResult<Option<ScalarValue>> {
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
            Ok(accumulator
                .checked_add(count)
                .map(|v| ScalarValue::Primitive(v.into())))
        }
        DType::Primitive(ptype, _) => {
            #[expect(dead_code, reason = "TODO(connor): good question")]
            let result = match_each_native_ptype!(
                ptype,
                unsigned: |T| { sum_integral::<u64>(scalar.as_primitive(), len, accumulator)?.map(|v| ScalarValue::Primitive(v.into())) },
                signed: |T| { sum_integral::<i64>(scalar.as_primitive(), len, accumulator)?.map(|v| ScalarValue::Primitive(v.into())) },
                floating: |T| { sum_float(scalar.as_primitive(), len, accumulator)?.map(|v| ScalarValue::Primitive(v.into())) }
            );
            Ok(result)
        }
        DType::Decimal(decimal_dtype, _) => {
            sum_decimal(scalar.as_decimal(), len, *decimal_dtype, accumulator)
        }
        DType::Extension(_) => {
            sum_scalar(&scalar.as_extension().to_storage_scalar(), len, accumulator)
        }
        dtype => vortex_bail!("Unsupported dtype for sum: {}", dtype),
    }
}

fn sum_decimal(
    decimal_scalar: DecimalScalar,
    array_len: usize,
    decimal_dtype: DecimalDType,
    accumulator: &Scalar,
) -> VortexResult<Option<ScalarValue>> {
    let result_dtype = Stat::Sum
        .dtype(&DType::Decimal(decimal_dtype, Nullability::Nullable))
        .vortex_expect("decimal supports sum");
    let result_decimal_type = result_dtype
        .as_decimal_opt()
        .vortex_expect("must be decimal");

    let Some(value) = decimal_scalar.decimal_value() else {
        // Null value: return null
        return Ok(None);
    };

    // Convert array_len to DecimalValue for multiplication.
    let len_value = DecimalValue::I256(i256::from_i128(array_len as i128));

    let Some(array_sum) = value
        .checked_mul(&len_value)
        .filter(|d| d.fits_in_precision(*result_decimal_type))
    else {
        return Ok(None);
    };

    // Add accumulator to array_sum.
    let initial_decimal = accumulator.as_decimal();
    let initial_dec_value = initial_decimal
        .decimal_value()
        .unwrap_or(DecimalValue::I256(i256::ZERO));

    let total = array_sum
        .checked_add(&initial_dec_value)
        .and_then(|result| {
            result
                .fits_in_precision(*result_decimal_type)
                .then_some(result)
        });
    match total {
        Some(result_value) => Ok(Some(ScalarValue::from(result_value))),
        None => Ok(None), // Overflow
    }
}

fn sum_integral<T>(
    primitive_scalar: PrimitiveScalar<'_>,
    array_len: usize,
    accumulator: &Scalar,
) -> VortexResult<Option<T>>
where
    T: NativePType + CheckedMul + CheckedAdd,
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
    let initial = accumulator
        .as_primitive()
        .as_::<f64>()
        .vortex_expect("cannot be null");
    let v = primitive_scalar
        .as_::<f64>()
        .vortex_expect("cannot be null");
    let len_f64: f64 = array_len.as_();

    Ok(Some(initial + v * len_f64))
}

register_kernel!(SumKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::compute::sum;
    use crate::compute::sum_with_accumulator;
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

    #[test]
    fn test_sum_float_non_multiply() {
        let acc = -2048669276050936500000000000f64;
        let array = ConstantArray::new(6.1811675e16f64, 25);
        let sum = sum_with_accumulator(array.as_ref(), &Scalar::primitive(acc, Nullable))
            .vortex_expect("operation should succeed in test");
        assert_eq!(
            f64::try_from(&sum).vortex_expect("operation should succeed in test"),
            -2048669274505644600000000000f64
        );
    }
}
