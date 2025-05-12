use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::ListEncoding;
use crate::compute::{FilterKernel, FilterKernelAdapter, arrow_filter_fn};
use crate::{ArrayRef, register_kernel};

impl FilterKernel for ListEncoding {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        arrow_filter_fn(array, mask)
    }
}

register_kernel!(FilterKernelAdapter(ListEncoding).lift());
