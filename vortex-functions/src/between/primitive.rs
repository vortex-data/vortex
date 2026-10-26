// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Primitive;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_native_ptype;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::between::BetweenKernel;
use crate::between::BetweenOptions;
use crate::between::StrictComparison;

impl BetweenKernel for Primitive {
    fn between(
        arr: ArrayView<'_, Primitive>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        // Note, we know that have checked before that the lower and upper bounds are not constant
        // null values

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        Ok(Some(match_each_native_ptype!(arr.ptype(), |P| {
            between_impl::<P>(
                arr,
                P::try_from(&lower)?,
                P::try_from(&upper)?,
                nullability,
                options,
                ctx,
            )
        })))
    }
}

fn between_impl<T: NativePType + Copy>(
    arr: ArrayView<'_, Primitive>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    match (options.lower_strict, options.upper_strict) {
        // Note: these comparisons are explicitly passed in to allow function impl inlining
        (StrictComparison::Strict, StrictComparison::Strict) => between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_lt,
            nullability,
            ctx,
        ),
        (StrictComparison::Strict, StrictComparison::NonStrict) => between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_le,
            nullability,
            ctx,
        ),
        (StrictComparison::NonStrict, StrictComparison::Strict) => between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_lt,
            nullability,
            ctx,
        ),
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_le,
            nullability,
            ctx,
        ),
    }
}

fn between_impl_<T>(
    arr: ArrayView<'_, Primitive>,
    lower: T,
    lower_fn: impl Fn(T, T) -> bool,
    upper: T,
    upper_fn: impl Fn(T, T) -> bool,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> ArrayRef
where
    T: NativePType + Copy,
{
    let slice = arr.as_slice::<T>();
    BoolArray::new(
        BitBuffer::collect_bool_multiversioned_in(
            slice.len(),
            |idx| {
                // We only iterate upto arr len and |arr| == |slice|.
                let i = unsafe { *slice.get_unchecked(idx) };
                lower_fn(lower, i) & upper_fn(i, upper)
            },
            ctx.allocator().clone(),
        ),
        arr.validity()
            .vortex_expect("validity should be derivable")
            .union_nullability(nullability),
    )
    .into_array()
}
