use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ListArray, ListEncoding};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl MaskKernel for ListEncoding {
    fn mask(&self, array: &ListArray, mask: &Mask) -> VortexResult<ArrayRef> {
        ListArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array.validity().mask(mask)?,
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ListEncoding).lift());
