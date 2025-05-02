use vortex_array::arrays::ConstantArray;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

use crate::SparseArray;

impl ArrayOperationsImpl for SparseArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let new_patches = self.patches().slice(start, stop)?;

        let Some(new_patches) = new_patches else {
            return Ok(ConstantArray::new(self.fill_scalar().clone(), stop - start).into_array());
        };

        // If the number of values in the sparse array matches the array length, then all
        // values are in fact patches, since patches are sorted this is the correct values.
        if new_patches.array_len() == new_patches.values().len() {
            return Ok(new_patches.into_values());
        }

        Ok(
            SparseArray::try_new_from_patches(new_patches, self.fill_scalar().clone())?
                .into_array(),
        )
    }
}

#[cfg(test)]
mod tests {

    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn slice_partially_invalid() {
        let values = buffer![0u64].into_array();
        let indices = buffer![0u8].into_array();

        let sparse = SparseArray::try_new(indices, values, 1000, 999u64.into()).unwrap();
        let sliced = sparse.slice(0, 1000).unwrap();
        let mut expected = vec![999u64; 1000];
        expected[0] = 0;

        let values = sliced.to_primitive().unwrap();
        assert_eq!(values.as_slice::<u64>(), expected);
    }
}
