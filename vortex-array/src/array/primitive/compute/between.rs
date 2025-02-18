use arrow_buffer::BooleanBuffer;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{BetweenFn, Operator};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray};

impl BetweenFn<PrimitiveArray> for PrimitiveEncoding {
    fn between(
        &self,
        arr: &PrimitiveArray,
        lower: &Array,
        lower_op: Operator,
        upper: &Array,
        upper_op: Operator,
    ) -> VortexResult<Option<Array>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        match_each_native_ptype!(arr.ptype(), |$P| {
            between_impl::<$P>(arr, $P::try_from(lower)?, lower_op, $P::try_from(upper)?, upper_op)
        })
        .map(Some)
    }
}

fn between_impl<T: NativePType + Copy>(
    arr: &PrimitiveArray,
    lower: T,
    lower_op: Operator,
    upper: T,
    upper_op: Operator,
) -> VortexResult<Array> {
    // match (lower_op, upper_op) {
    //     (Operator::Lte, Operator)
    // }

    let lower_fn = lower_op.to_fn();
    let upper_fn = upper_op.to_fn();

    let slice = arr.as_slice::<T>();
    Ok(
        BoolArray::from(BooleanBuffer::collect_bool(arr.len(), |idx| {
            let i = *unsafe { slice.get_unchecked(idx) };
            lower_fn(lower, i) & upper_fn(i, upper)
        }))
        .into_array(),
    )
}
