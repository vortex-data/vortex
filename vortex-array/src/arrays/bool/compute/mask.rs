use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::MaskFn;
use crate::{Array, ArrayRef};

impl MaskFn<&BoolArray> for BoolEncoding {
    fn mask(&self, array: &BoolArray, mask: Mask) -> VortexResult<ArrayRef> {
        Ok(BoolArray::new(
            array.boolean_buffer().clone(),
            array.validity().mask(&mask)?,
        )
        .into_array())
    }
}
