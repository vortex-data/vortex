use arrow_buffer::BooleanBuffer;
use vortex_dtype::{NativePType, Nullability, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::arrays::{BoolArray, PrimitiveArray, PrimitiveVTable};
use crate::compute::{BetweenKernel, BetweenKernelAdapter, BetweenOptions, StrictComparison};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

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
            arr.dtype.nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        Ok(Some(match_each_native_ptype!(arr.ptype(), |P| {
            between_impl::<P>(
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

fn between_impl<T: NativePType + Copy>(
    arr: &PrimitiveArray,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
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
        ),
        (StrictComparison::Strict, StrictComparison::NonStrict) => between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_le,
            nullability,
        ),
        (StrictComparison::NonStrict, StrictComparison::Strict) => between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_lt,
            nullability,
        ),
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_le,
            nullability,
        ),
    }
}

fn between_impl_<T>(
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
