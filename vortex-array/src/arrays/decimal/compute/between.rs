// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenKernel;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;

impl BetweenKernel for Decimal {
    fn between(
        arr: ArrayView<'_, Decimal>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
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
            between_unpack::<D>(arr, lower, upper, nullability, options)
        })
    }
}

fn between_unpack<T: NativeDecimalType>(
    arr: ArrayView<'_, Decimal>,
    lower: Scalar,
    upper: Scalar,
    nullability: Nullability,
    options: &BetweenOptions,
) -> VortexResult<Option<ArrayRef>> {
    let Some(lower_dv) = lower.as_decimal().decimal_value() else {
        // Null lower bound — fall back to canonical path.
        return Ok(None);
    };
    let Some(upper_dv) = upper.as_decimal().decimal_value() else {
        // Null upper bound — fall back to canonical path.
        return Ok(None);
    };

    // Try to cast the bound scalar to the array's storage type T.
    //
    // If the cast fails, the bound's value is outside [T::MIN, T::MAX].  For all signed
    // NativeDecimalType implementations the minimum is negative and the maximum is positive, so
    // we can determine the direction of overflow from the sign of the bound:
    //   • non-negative and doesn't fit in T  ⟹  value > T::MAX
    //   • negative and doesn't fit in T      ⟹  value < T::MIN
    //
    // From the direction we can answer the comparison immediately:
    //   lower > T::MAX: no array value (≤ T::MAX) satisfies lower ≤ value  → all-false
    //   lower < T::MIN: every array value (≥ T::MIN) satisfies lower ≤ value → no lower constraint
    //   upper > T::MAX: every array value (≤ T::MAX) satisfies value ≤ upper → no upper constraint
    //   upper < T::MIN: no array value (≥ T::MIN) satisfies value ≤ upper   → all-false
    //
    // Both the strict and non-strict forms lead to the same conclusion because the overflow is
    // by at least one integer, so no boundary element can make the strict form differ.
    let lower_value: Option<T> = match lower_dv.cast::<T>() {
        Some(v) => Some(v),
        None => {
            if lower_dv.as_i256() >= i256::ZERO {
                return Ok(Some(
                    ConstantArray::new(Scalar::bool(false, nullability), arr.len()).into_array(),
                ));
            }
            None
        }
    };

    let upper_value: Option<T> = match upper_dv.cast::<T>() {
        Some(v) => Some(v),
        None => {
            if upper_dv.as_i256() < i256::ZERO {
                return Ok(Some(
                    ConstantArray::new(Scalar::bool(false, nullability), arr.len()).into_array(),
                ));
            }
            None
        }
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

fn between_impl<T: NativeDecimalType>(
    arr: ArrayView<'_, Decimal>,
    lower: Option<T>,
    upper: Option<T>,
    nullability: Nullability,
    lower_op: impl Fn(T, T) -> bool,
    upper_op: impl Fn(T, T) -> bool,
) -> ArrayRef {
    let buffer = arr.buffer::<T>();
    BoolArray::new(
        BitBuffer::collect_bool(buffer.len(), |idx| {
            let value = buffer[idx];
            lower.map_or(true, |l| lower_op(l, value)) & upper.map_or(true, |u| upper_op(value, u))
        }),
        arr.validity()
            .vortex_expect("validity should be derivable")
            .union_nullability(nullability),
    )
    .into_array()
}
