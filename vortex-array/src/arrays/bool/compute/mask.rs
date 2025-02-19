use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::MaskFn;
use crate::{Array, IntoArray};

impl MaskFn<BoolArray> for BoolEncoding {
    fn mask(&self, array: &BoolArray, mask: Mask) -> VortexResult<Array> {
        BoolArray::try_new(array.boolean_buffer(), array.validity().mask(&mask)?)
            .map(IntoArray::into_array)
    }
}
