use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

impl IsSortedKernel for VarBinViewEncoding {
    fn is_sorted(&self, array: &VarBinViewArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted())
    }

    fn is_strict_sorted(&self, array: &VarBinViewArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinViewEncoding).lift());
