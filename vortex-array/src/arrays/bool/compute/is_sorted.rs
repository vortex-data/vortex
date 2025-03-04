use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::{IsSortedFn, IteratorExt};

impl IsSortedFn<&BoolArray> for BoolEncoding {
    fn is_sorted(&self, array: &BoolArray, strict: bool) -> VortexResult<bool> {
        match array.validity_mask()? {
            Mask::AllFalse(_) => Ok(!strict),
            Mask::AllTrue(_) => Ok(array
                .boolean_buffer()
                .iter()
                .is_sorted_with_strictness(strict)),
            Mask::Values(mask_values) => {
                let set_indices = mask_values.boolean_buffer().set_indices();
                let values = array.boolean_buffer();
                let values_iter = set_indices.map(|idx|
                    // Safety:
                    // All idxs are in-bounds for the array.
                    unsafe {
                        values.value_unchecked(idx)
                    });

                Ok(values_iter.is_sorted_with_strictness(strict))
            }
        }
    }
}
