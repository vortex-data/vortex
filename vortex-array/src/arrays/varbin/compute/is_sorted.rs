use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

impl IsSortedKernel for VarBinEncoding {
    fn is_sorted(&self, array: &VarBinArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_sorted())
    }

    fn is_strict_sorted(&self, array: &VarBinArray) -> VortexResult<bool> {
        array.with_iterator(|bytes_iter| bytes_iter.is_strict_sorted())
    }
}

register_kernel!(IsSortedKernelAdapter(VarBinEncoding).lift());
