// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
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
            arr.dtype.nullability() | lower.dtype().nullability() | upper.dtype().nullability();

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

fn between_impl<T: NativeDecimalType>(
    arr: ArrayView<'_, Decimal>,
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
        arr.validity().union_nullability(nullability),
    )
    .into_array()
}
