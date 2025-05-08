use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::{FilterKernel, FilterKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl FilterKernel for ConstantEncoding {
    fn filter(&self, array: &ConstantArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().clone(), mask.true_count()).into_array())
    }
}

register_kernel!(FilterKernelAdapter(ConstantEncoding).lift());
