// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::compute::BetweenKernel;
use crate::compute::BetweenKernelAdapter;
use crate::compute::BetweenOptions;
use crate::compute::StrictComparison;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl BetweenKernel for DecimalVTable {
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

        match_each_decimal_value_type!(arr.values_type(), |D| {
            between_unpack::<D>(arr, lower, upper, nullability, options)
        })
    }
}

fn between_unpack<T: NativeDecimalType>(
    arr: &DecimalArray,
    lower: Scalar,
    upper: Scalar,
    nullability: Nullability,
    options: &BetweenOptions,
) -> VortexResult<Option<ArrayRef>> {
    let Some(lower_value) = lower
        .as_decimal()
        .decimal_value()
        .and_then(|v| v.cast::<T>())
    else {
        vortex_bail!(
            "invalid lower bound Scalar: {lower}, expected {:?}",
            T::DECIMAL_TYPE
        )
    };
    let Some(upper_value) = upper
        .as_decimal()
        .decimal_value()
        .and_then(|v| v.cast::<T>())
    else {
        vortex_bail!(
            "invalid upper bound Scalar: {upper}, expected {:?}",
            T::DECIMAL_TYPE
        )
    };

    let lower_op = match options.lower_strict {
        StrictComparison::Strict => |a, b| a < b,
        StrictComparison::NonStrict => |a, b| a <= b,
    };

    let upper_op = match options.upper_strict {
        StrictComparison::Strict => |a, b| a < b,
        StrictComparison::NonStrict => |a, b| a <= b,
    };

    Ok(Some(between_impl::<T>(
        arr,
        lower_value,
        upper_value,
        nullability,
        lower_op,
        upper_op,
    )))
}

register_kernel!(BetweenKernelAdapter(DecimalVTable).lift());

fn between_impl<T: NativeDecimalType>(
    arr: &DecimalArray,
    lower: T,
    upper: T,
    nullability: Nullability,
    lower_op: impl Fn(T, T) -> bool,
    upper_op: impl Fn(T, T) -> bool,
) -> ArrayRef {
    let buffer = arr.buffer::<T>();
    BoolArray::new(
        BitBuffer::collect_bool(buffer.len(), |idx| {
            let value = buffer[idx];
            lower_op(lower, value) & upper_op(value, upper)
        }),
        arr.validity().clone().union_nullability(nullability),
    )
    .into_array()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::Nullability;
    use vortex_scalar::DecimalValue;
    use vortex_scalar::Scalar;

    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::compute::BetweenOptions;
    use crate::compute::StrictComparison;
    use crate::compute::between;
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
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap();
        assert_arrays_eq!(
            between_strict,
            BoolArray::from_iter([false, true, true, true])
        );

        // Non-strict lower bound, strict upper bound
        let between_strict = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        )
        .unwrap();
        assert_arrays_eq!(
            between_strict,
            BoolArray::from_iter([true, true, true, false])
        );
    }
}
