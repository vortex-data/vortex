// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical implementations of the Between compute kernel.
//!
//! These implementations provide optimized between operations for core array types.

use arrow_buffer::BooleanBuffer;
use vortex_dtype::{NativePType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{NativeDecimalType, Scalar, match_each_decimal_value_type};

use crate::arrays::{BoolArray, PrimitiveArray, PrimitiveVTable, DecimalArray, DecimalVTable};
use crate::compute::{BetweenKernel, BetweenKernelAdapter, BetweenOptions, StrictComparison};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

/// Implementation of Between kernel for PrimitiveArray
impl BetweenKernel for PrimitiveVTable {
    fn between(
        &self,
        arr: &PrimitiveArray,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        // Note, we know that have checked before that the lower and upper bounds are not constant
        // null values

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        Ok(Some(match_each_native_ptype!(arr.ptype(), |P| {
            primitive_between_impl::<P>(
                arr,
                P::try_from(lower)?,
                P::try_from(upper)?,
                nullability,
                options,
            )
        })))
    }
}

register_kernel!(BetweenKernelAdapter(PrimitiveVTable).lift());

fn primitive_between_impl<T: NativePType + Copy>(
    arr: &PrimitiveArray,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
) -> ArrayRef {
    match (options.lower_strict, options.upper_strict) {
        // Note: these comparisons are explicitly passed in to allow function impl inlining
        (StrictComparison::Strict, StrictComparison::Strict) => primitive_between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_lt,
            nullability,
        ),
        (StrictComparison::Strict, StrictComparison::NonStrict) => primitive_between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_le,
            nullability,
        ),
        (StrictComparison::NonStrict, StrictComparison::Strict) => primitive_between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_lt,
            nullability,
        ),
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => primitive_between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_le,
            nullability,
        ),
    }
}

fn primitive_between_impl_<T>(
    arr: &PrimitiveArray,
    lower: T,
    lower_fn: impl Fn(T, T) -> bool,
    upper: T,
    upper_fn: impl Fn(T, T) -> bool,
    nullability: Nullability,
) -> ArrayRef
where
    T: NativePType + Copy,
{
    let slice = arr.as_slice::<T>();
    BoolArray::new(
        BooleanBuffer::collect_bool(slice.len(), |idx| {
            // We only iterate upto arr len and |arr| == |slice|.
            let i = unsafe { *slice.get_unchecked(idx) };
            lower_fn(lower, i) & upper_fn(i, upper)
        }),
        arr.validity().clone().union_nullability(nullability),
    )
    .into_array()
}

/// Implementation of Between kernel for DecimalArray
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
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        match_each_decimal_value_type!(arr.values_type(), |D| {
            decimal_between_unpack::<D>(arr, lower, upper, nullability, options)
        })
    }
}

fn decimal_between_unpack<T: NativeDecimalType>(
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
            T::VALUES_TYPE
        )
    };
    let Some(upper_value) = upper
        .as_decimal()
        .decimal_value()
        .and_then(|v| v.cast::<T>())
    else {
        vortex_bail!(
            "invalid upper bound Scalar: {upper}, expected {:?}",
            T::VALUES_TYPE
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

    Ok(Some(decimal_between_impl::<T>(
        arr,
        lower_value,
        upper_value,
        nullability,
        lower_op,
        upper_op,
    )))
}

register_kernel!(BetweenKernelAdapter(DecimalVTable).lift());

fn decimal_between_impl<T: NativeDecimalType>(
    arr: &DecimalArray,
    lower: T,
    upper: T,
    nullability: Nullability,
    lower_op: impl Fn(T, T) -> bool,
    upper_op: impl Fn(T, T) -> bool,
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
    fn test_decimal_between() {
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
        assert_eq!(bool_to_vec(&between_strict), vec![false, true, true, true]);

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