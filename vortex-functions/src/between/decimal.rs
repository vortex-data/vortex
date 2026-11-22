// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Decimal;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::scalar::DecimalValue;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::between::BetweenKernel;
use crate::between::BetweenOptions;
use crate::between::StrictComparison;

impl BetweenKernel for Decimal {
    fn between(
        arr: ArrayView<'_, Decimal>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        if !arr.dtype().eq_ignore_nullability(lower.dtype())
            || !arr.dtype().eq_ignore_nullability(upper.dtype())
        {
            return Ok(None);
        }
        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        let (Some(lower), Some(upper)) = (
            lower.as_decimal().decimal_value(),
            upper.as_decimal().decimal_value(),
        ) else {
            // A null bound needs the Kleene AND of the two comparisons.
            return Ok(None);
        };

        let bits = match_each_decimal_value_type!(arr.values_type(), |T| {
            between_values(
                &arr.buffer::<T>(),
                lower,
                upper,
                options,
                ctx.allocator().clone(),
            )
        });
        let validity = arr.validity()?.union_nullability(nullability);

        Ok(Some(BoolArray::new(bits, validity).into_array()))
    }
}

fn between_values<T: NativeDecimalType>(
    values: &[T],
    lower: DecimalValue,
    upper: DecimalValue,
    options: &BetweenOptions,
    allocator: BufferAllocatorRef,
) -> BitBuffer {
    let lower = lower
        .cast::<T>()
        .vortex_expect("a decimal bound fits the canonical width of its precision");
    let upper = upper
        .cast::<T>()
        .vortex_expect("a decimal bound fits the canonical width of its precision");

    match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => {
            between_bits(values, |v| (lower < v) & (v < upper), allocator)
        }
        (StrictComparison::Strict, StrictComparison::NonStrict) => {
            between_bits(values, |v| (lower < v) & (v <= upper), allocator)
        }
        (StrictComparison::NonStrict, StrictComparison::Strict) => {
            between_bits(values, |v| (lower <= v) & (v < upper), allocator)
        }
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => {
            between_bits(values, |v| (lower <= v) & (v <= upper), allocator)
        }
    }
}

fn between_bits<T: Copy>(
    values: &[T],
    predicate: impl Fn(T) -> bool,
    allocator: BufferAllocatorRef,
) -> BitBuffer {
    BitBuffer::collect_bool_multiversioned_in(
        values.len(),
        |index| {
            // SAFETY: The collector visits only indices in 0..values.len().
            let value = unsafe { *values.get_unchecked(index) };
            predicate(value)
        },
        allocator,
    )
}
