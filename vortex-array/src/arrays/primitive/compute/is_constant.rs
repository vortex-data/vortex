use vortex_dtype::{NativePType, match_each_native_ptype};

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::IsConstantFn;
use crate::variants::PrimitiveArrayTrait;

impl IsConstantFn<&PrimitiveArray> for PrimitiveEncoding {
    fn is_constant(&self, array: &PrimitiveArray) -> vortex_error::VortexResult<Option<bool>> {
        let is_constant = match_each_native_ptype!(array.ptype(), |$P| {
            compute_is_constant(array.as_slice::<$P>())
        });

        Ok(Some(is_constant))
    }
}

// Assumes there's at least 1 value in the slice, which is an invariant of the entry level function.
fn compute_is_constant<T: NativePType>(values: &[T]) -> bool {
    let first_value = values[0];

    for value in &values[1..] {
        if *value != first_value {
            return false;
        }
    }

    true
}
