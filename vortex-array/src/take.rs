use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::array::PrimitiveArray;
use crate::variants::PrimitiveArrayTrait as _;

pub fn take(values: &PrimitiveArray, indices: &PrimitiveArray) -> VortexResult<PrimitiveArray> {
    let new_validity = values.validity().take(indices.as_ref())?;
    match_each_native_ptype!(values.ptype(), |$V| {
        let values = values.maybe_null_slice::<$V>();
        let new_values = match_each_integer_ptype!(indices.ptype(), |$I| {
            indices
                .maybe_null_slice::<$I>()
                .iter()
                .cloned()
                .map(|idx| values[idx as usize])
                .collect()
        });
        Ok(PrimitiveArray::from_vec(new_values, new_validity))
    })
}

pub unsafe fn take_unchecked(
    values: &PrimitiveArray,
    indices: &PrimitiveArray,
) -> VortexResult<PrimitiveArray> {
    let new_validity = unsafe { values.validity().take_unchecked(indices.as_ref())? };
    match_each_native_ptype!(values.ptype(), |$V| {
        let values = values.maybe_null_slice::<$V>();
        let new_values = match_each_integer_ptype!(indices.ptype(), |$I| {
            indices
                .maybe_null_slice::<$I>()
                .iter()
                .cloned()
                .map(|idx| unsafe { *values.get_unchecked(idx as usize) })
                .collect()
        });
        Ok(PrimitiveArray::from_vec(new_values, new_validity))
    })
}
