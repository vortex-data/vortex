use vortex_array::compute::{invert, InvertFn};
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_error::VortexResult;

use crate::{SparseArray, SparseEncoding};

impl InvertFn<SparseArray> for SparseEncoding {
    fn invert(&self, array: &SparseArray) -> VortexResult<ArrayData> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array.patches().map_values(|values| invert(&values))?;
        SparseArray::try_new_from_patches(
            inverted_patches,
            array.len(),
            array.indices_offset(),
            inverted_fill,
        )
        .map(|a| a.into_array())
    }
}
