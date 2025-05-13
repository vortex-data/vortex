use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::VarBinViewVTable;
use crate::compute::{FilterKernel, FilterKernelAdapter, arrow_filter_fn};
use crate::{ArrayRef, register_kernel};

impl FilterKernel for VarBinViewVTable {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        arrow_filter_fn(array.as_ref(), mask)
    }
}

register_kernel!(FilterKernelAdapter(VarBinViewVTable).lift());
