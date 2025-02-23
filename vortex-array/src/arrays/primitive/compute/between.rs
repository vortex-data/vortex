use arrow_buffer::BooleanBuffer;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::arrays::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{BetweenFn, BetweenOptions};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, IntoArray};

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
    let lower_fn = options.lower_strict.to_operator().to_fn();
    let upper_fn = options.upper_strict.to_operator().to_fn();

    let slice = arr.as_slice::<T>();
    BoolArray::new(
        BooleanBuffer::collect_bool(arr.len(), |idx| {
            // We only iterate upto arr len and |arr| == |slice|.
            let i = *unsafe { slice.get_unchecked(idx) };
            lower_fn(lower, i) & upper_fn(i, upper)
        }),
        arr.validity().clone(),
    )
    .into_array()
}
