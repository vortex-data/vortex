use arrow_buffer::BooleanBuffer;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::arrays::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{BetweenFn, BetweenOptions};
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
    let lower_fn = options.lower_strict.to_operator().to_fn();
    let upper_fn = options.upper_strict.to_operator().to_fn();

    let slice = arr.as_slice::<T>();
    BoolArray::try_new(
        BooleanBuffer::collect_bool(arr.len(), |idx| {
            // We only iterate upto arr len and |arr| == |slice|.
            let i = *unsafe { slice.get_unchecked(idx) };
            lower_fn(lower, i) & upper_fn(i, upper)
        }),
        arr.validity(),
    )
    .map(BoolArray::into_array)
}
