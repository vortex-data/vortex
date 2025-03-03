use arrow_buffer::BooleanBuffer;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::arrays::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{BetweenFn, BetweenOptions, StrictComparison};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef};

impl BetweenFn<&PrimitiveArray> for PrimitiveEncoding {
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

        Ok(Some(match_each_native_ptype!(arr.ptype(), |$P| {
            between_impl::<$P>(arr, $P::try_from(lower)?, $P::try_from(upper)?, options)
        })))
    }
}

fn between_impl<T: NativePType + Copy>(
    arr: &PrimitiveArray,
    lower: T,
    upper: T,
    options: &BetweenOptions,
) -> ArrayRef {
    match (options.lower_strict, options.upper_strict) {
        // Note: these comparisons are explicitly passed in to allow function impl inlining
        (StrictComparison::Strict, StrictComparison::Strict) => {
            between_impl_(arr, lower, NativePType::is_lt, upper, NativePType::is_lt)
        }
        (StrictComparison::Strict, StrictComparison::NonStrict) => {
            between_impl_(arr, lower, NativePType::is_lt, upper, NativePType::is_le)
        }
        (StrictComparison::NonStrict, StrictComparison::Strict) => {
            between_impl_(arr, lower, NativePType::is_le, upper, NativePType::is_lt)
        }
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => {
            between_impl_(arr, lower, NativePType::is_le, upper, NativePType::is_le)
        }
    }
}

fn between_impl_<T>(
    arr: &PrimitiveArray,
    lower: T,
    lower_fn: impl Fn(T, T) -> bool,
    upper: T,
    upper_fn: impl Fn(T, T) -> bool,
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
        arr.validity().clone(),
    )
    .into_array()
}
