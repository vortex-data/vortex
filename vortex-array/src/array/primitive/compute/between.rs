use arrow_buffer::BooleanBuffer;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::BetweenFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray};

impl BetweenFn<PrimitiveArray> for PrimitiveEncoding {
    fn between(
        &self,
        arr: &PrimitiveArray,
        lower: &Array,
        upper: &Array,
    ) -> VortexResult<Option<Array>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        // if ptype.is_int() {
        match_each_native_ptype!(arr.ptype(), |$P| {
            between_impl::<$P>(arr, $P::try_from(lower)?, $P::try_from(upper)?)
        })
        .map(Some)
        // } else if ptype.is_float() {
        //     match_each_float_ptype!(arr.ptype(), |$P| {
        //         between_float_impl::<$P>(arr, $P::try_from(lower)?, $P::try_from(upper)?)
        //     })
        //     .map(Some)
        // } else {
        //     vortex_panic!("not impl")
        // }
    }
}

// fn between_impl<T: NativePType>(arr: &PrimitiveArray, lower: T, upper: T) -> VortexResult<Array> {
//     let slice = arr.as_slice::<T>();
//
//     Ok(BoolArray::from(collect_bool(arr.len(), |idx| {
//         let i = unsafe { slice.get_unchecked(idx) };
//         i >= &lower && i <= &upper
//     }))
//     .into_array())
// }

fn between_impl<T: NativePType + Copy>(
    arr: &PrimitiveArray,
    lower: T,
    upper: T,
) -> VortexResult<Array> {
    let slice = arr.as_slice::<T>();
    let min1 = T::from_usize(2).unwrap();
    let max2 = T::from_usize(400).unwrap();

    let bool_buf = BooleanBuffer::collect_bool(arr.len(), |idx| {
        let i = *unsafe { slice.get_unchecked(idx) };
        // ((i > min1) & (i < max2)) as bool | ((i >= lower) & (i <= upper)) as bool
        ((i > min1) && (i < max2)) | ((i >= lower) && (i <= upper))
    });
    Ok(BoolArray::from(bool_buf).into_array())
}
