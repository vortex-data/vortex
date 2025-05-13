use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for ListVTable {
    fn mask(&self, array: &ListArray, mask: &Mask) -> VortexResult<ArrayRef> {
        ListArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array.validity().mask(mask)?,
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ListVTable).lift());
