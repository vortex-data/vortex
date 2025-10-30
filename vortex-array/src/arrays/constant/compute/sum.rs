// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::{CheckedMul, ToPrimitive};
use vortex_dtype::{DType, DecimalDType, NativePType, Nullability, i256, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{
    DecimalScalar, DecimalValue, FromPrimitiveOrF16, PrimitiveScalar, Scalar, ScalarValue,
};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;
use crate::stats::Stat;

impl SumKernel for ConstantVTable {
    fn sum(&self, array: &ConstantArray) -> VortexResult<Scalar> {
        // Compute the expected dtype of the sum.
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;

        let sum_value = sum_scalar(array.scalar(), array.len())?;
        Ok(Scalar::new(sum_dtype, sum_value))
    }
}

fn sum_scalar(scalar: &Scalar, len: usize) -> VortexResult<ScalarValue> {
    match scalar.dtype() {
        DType::Bool(_) => Ok(ScalarValue::from(match scalar.as_bool().value() {
            None => unreachable!("Handled before reaching this point"),
            Some(false) => 0u64,
            Some(true) => len as u64,
        })),
        DType::Primitive(ptype, _) => Ok(match_each_native_ptype!(
            ptype,
            unsigned: |T| { sum_integral::<u64>(scalar.as_primitive(), len)?.into() },
            signed: |T| { sum_integral::<i64>(scalar.as_primitive(), len)?.into() },
            floating: |T| { sum_float(scalar.as_primitive(), len)?.into() }
        )),
        DType::Decimal(decimal_dtype, _) => sum_decimal(scalar.as_decimal(), len, *decimal_dtype),
        DType::Extension(_) => sum_scalar(&scalar.as_extension().storage(), len),
        dtype => vortex_bail!("Unsupported dtype for sum: {}", dtype),
    }
}

fn sum_decimal(
    decimal_scalar: DecimalScalar,
    array_len: usize,
    decimal_dtype: DecimalDType,
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
    let sum = value.checked_mul(&len_value).and_then(|result| {
        // Check if result fits in the precision
        result
            .fits_in_precision(*result_decimal_type)
            .unwrap_or(false)
            .then_some(result)
    });

    match sum {
        Some(result_value) => Ok(ScalarValue::from(result_value)),
        None => Ok(ScalarValue::null()), // Overflow
    }
}

fn sum_integral<T>(
    primitive_scalar: PrimitiveScalar<'_>,
    array_len: usize,
) -> VortexResult<Option<T>>
where
    T: FromPrimitiveOrF16 + NativePType + CheckedMul,
    Scalar: From<Option<T>>,
{
    let v = primitive_scalar.as_::<T>();
    let array_len =
        T::from(array_len).ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;
    let sum = v.and_then(|v| v.checked_mul(&array_len));

    Ok(sum)
}

fn sum_float(primitive_scalar: PrimitiveScalar<'_>, array_len: usize) -> VortexResult<Option<f64>> {
    let v = primitive_scalar.as_::<f64>();
    let array_len = array_len
        .to_f64()
        .ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;

    Ok(v.map(|v| v * array_len))
}

register_kernel!(SumKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, DecimalDType, Nullability, PType};
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
        let array = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable)),
            10,
        )
        .into_array();
        let result = sum(&array).unwrap();
        assert!(result.is_null());
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
        let array =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 10).into_array();
        let result = sum(&array).unwrap();
        assert!(result.is_null());
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
            Some(DecimalValue::I256(vortex_scalar::i256::from_i128(500)))
        );
        assert_eq!(result.dtype(), &Stat::Sum.dtype(array.dtype()).unwrap());
    }

    #[test]
    fn test_sum_decimal_null() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable)),
            10,
        )
        .into_array();

        let result = sum(&array).unwrap();
        assert!(result.is_null());
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
            Some(DecimalValue::I256(vortex_scalar::i256::from_i128(
                99_999_999_900
            )))
        );
    }
}
