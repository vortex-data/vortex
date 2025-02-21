use arrow_buffer::BooleanBuffer;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::arrays::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{BetweenFn, BetweenOptions, Lt, Lte, OperatorImpl, StrictComparison};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray};

impl BetweenFn<PrimitiveArray> for PrimitiveEncoding {
    fn between(
        &self,
        arr: &PrimitiveArray,
        lower: &Array,
        upper: &Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<Array>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        match_each_native_ptype!(arr.ptype(), |$P| {
            between_impl::<$P>(arr, $P::try_from(lower)?, $P::try_from(upper)?, options)
        })
        .map(Some)
    }
}

fn between_impl<T: NativePType + Copy>(
    arr: &PrimitiveArray,
    lower: T,
    upper: T,
    options: &BetweenOptions,
) -> VortexResult<Array> {
    match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => {
            between_impl_::<Lt, Lt, _>(arr, lower, upper)
        }
        (StrictComparison::Strict, StrictComparison::NonStrict) => {
            between_impl_::<Lt, Lte, _>(arr, lower, upper)
        }
        (StrictComparison::NonStrict, StrictComparison::Strict) => {
            between_impl_::<Lte, Lt, _>(arr, lower, upper)
        }
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => {
            between_impl_::<Lte, Lte, _>(arr, lower, upper)
        }
    }
}

fn between_impl_<Lower, Upper, T>(arr: &PrimitiveArray, lower: T, upper: T) -> VortexResult<Array>
where
    T: NativePType + Copy,
    Lower: OperatorImpl<T>,
    Upper: OperatorImpl<T>,
{
    let slice = arr.as_slice::<T>();
    BoolArray::try_new(
        BooleanBuffer::collect_bool(slice.len(), |idx| {
            // We only iterate upto arr len and |arr| == |slice|.
            let i = *unsafe { slice.get_unchecked(idx) };
            Lower::FN_(lower, i) & Upper::FN_(i, upper)
        }),
        arr.validity(),
    )
    .map(BoolArray::into_array)
}
