use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl MaskKernel for BoolEncoding {
    fn mask(&self, array: &BoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            BoolArray::new(array.boolean_buffer().clone(), array.validity().mask(mask)?)
                .into_array(),
        )
    }
}

register_kernel!(MaskKernelAdapter(BoolEncoding).lift());
