use vortex_array::compute::{InvertFn, invert};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;

use crate::{SparseArray, SparseEncoding};

impl InvertFn<&SparseArray> for SparseEncoding {
    fn invert(&self, array: &SparseArray) -> VortexResult<ArrayRef> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array
            .patches()
            .clone()
            .map_values(|values| invert(&values))?;
        SparseArray::try_new_from_patches(inverted_patches, inverted_fill).map(|a| a.into_array())
    }
}
