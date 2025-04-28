use arrow_buffer::BooleanBuffer;
use vortex_dtype::Nullability;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{DecimalValue, i256};

use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{BoolArray, DecimalArray, DecimalEncoding, NativeDecimalType};
use crate::compute::{BetweenKernel, BetweenKernelAdapter, BetweenOptions, StrictComparison};
use crate::{Array, ArrayRef, register_kernel};

impl BetweenKernel for DecimalEncoding {
    // Determine if the values are between the lower and upper bounds
    fn between(
        &self,
        arr: &DecimalArray,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // NOTE: We know that the precision and scale were already checked to be equal by the main
        // `between` entrypoint function.

        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        // NOTE: we know that have checked before that the lower and upper bounds are not all null.
        let nullability =
            arr.dtype.nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        match arr.values_type {
            DecimalValueType::I128 => {
                let Some(DecimalValue::I128(lower_128)) = *lower.as_decimal().decimal_value()
                else {
                    vortex_bail!("invalid lower bound Scalar: {lower}");
                };
                let Some(DecimalValue::I128(upper_128)) = *upper.as_decimal().decimal_value()
                else {
                    vortex_bail!("invalid upper bound Scalar: {upper}");
                };

                let lower_op = match options.lower_strict {
                    StrictComparison::Strict => i128_lt_i128,
                    StrictComparison::NonStrict => i128_lte_i128,
                };

                let upper_op = match options.upper_strict {
                    StrictComparison::Strict => i128_lt_i128,
                    StrictComparison::NonStrict => i128_lte_i128,
                };

                Ok(Some(between_impl::<i128>(
                    arr,
                    lower_128,
                    upper_128,
                    nullability,
                    lower_op,
                    upper_op,
                )))
            }
            DecimalValueType::I256 => {
                let Some(DecimalValue::I256(lower_256)) = *lower.as_decimal().decimal_value()
                else {
                    vortex_bail!("invalid lower bound Scalar: {lower}");
                };
                let Some(DecimalValue::I256(upper_256)) = *upper.as_decimal().decimal_value()
                else {
                    vortex_bail!("invalid upper bound Scalar: {upper}");
                };

                let lower_op = match options.lower_strict {
                    StrictComparison::Strict => i256_lt_i256,
                    StrictComparison::NonStrict => i256_lte_i256,
                };

                let upper_op = match options.upper_strict {
                    StrictComparison::Strict => i256_lt_i256,
                    StrictComparison::NonStrict => i256_lte_i256,
                };

                Ok(Some(between_impl::<i256>(
                    arr,
                    lower_256,
                    upper_256,
                    nullability,
                    lower_op,
                    upper_op,
                )))
            }
        }
    }
}

register_kernel!(BetweenKernelAdapter(DecimalEncoding).lift());

fn between_impl<T: NativeDecimalType>(
    arr: &DecimalArray,
    lower: T,
    upper: T,
    nullability: Nullability,
    lower_op: fn(T, T) -> bool,
    upper_op: fn(T, T) -> bool,
) -> ArrayRef {
    let buffer = arr.buffer::<T>();
    BoolArray::new(
        BooleanBuffer::collect_bool(buffer.len(), |idx| {
            let value = buffer[idx];
            lower_op(lower, value) & upper_op(value, upper)
        }),
        arr.validity().clone().union_nullability(nullability),
    )
    .into_array()
}

#[inline]
const fn i128_lt_i128(a: i128, b: i128) -> bool {
    a < b
}

#[inline]
const fn i128_lte_i128(a: i128, b: i128) -> bool {
    a <= b
}

#[inline]
fn i256_lt_i256(a: i256, b: i256) -> bool {
    a < b
}

#[inline]
fn i256_lte_i256(a: i256, b: i256) -> bool {
    a <= b
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::{DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::Array;
    use crate::arrays::{ConstantArray, DecimalArray};
    use crate::compute::{BetweenOptions, StrictComparison, between};
    use crate::validity::Validity;

    #[test]
    fn test_between() {
        let values = buffer![100i128, 200i128, 300i128, 400i128];
        let decimal_type = DecimalDType::new(3, 2);
        let array = DecimalArray::new(values, decimal_type, Validity::NonNullable);

        let lower = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(100i128),
                decimal_type,
                Nullability::NonNullable,
            ),
            array.len(),
        );
        let upper = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(400i128),
                decimal_type,
                Nullability::NonNullable,
            ),
            array.len(),
        );

        // Strict lower bound, non-strict upper bound
        let between_strict = between(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap();
        assert_eq!(bool_to_vec(&between_strict), vec![false, true, true, true]);

        // Non-strict lower bound, strict upper bound
        let between_strict = between(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        )
        .unwrap();
        assert_eq!(bool_to_vec(&between_strict), vec![true, true, true, false]);
    }

    fn bool_to_vec(array: &dyn Array) -> Vec<bool> {
        array
            .to_canonical()
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .iter()
            .collect()
    }
}
