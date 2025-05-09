use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for BoolVTable {
    fn mask(&self, array: &BoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            BoolArray::new(array.boolean_buffer().clone(), array.validity().mask(mask)?)
                .into_array(),
        )
    }
}

register_kernel!(MaskKernelAdapter(BoolVTable).lift());
