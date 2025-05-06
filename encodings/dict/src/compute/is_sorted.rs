use vortex_array::compute::{IsSortedKernel, IsSortedKernelAdapter, is_sorted, is_strict_sorted};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl IsSortedKernel for DictEncoding {
    fn is_sorted(&self, array: &DictArray) -> VortexResult<bool> {
        // TODO(ngates): we should change these kernels to return Option<bool> to allow for "unknown"
        let is_sorted = is_sorted(array.values())? && is_sorted(array.codes())?;
        Ok(is_sorted)
    }

    fn is_strict_sorted(&self, array: &DictArray) -> VortexResult<bool> {
        let is_sorted = is_strict_sorted(array.values())? && is_strict_sorted(array.codes())?;
        Ok(is_sorted)
    }
}

register_kernel!(IsSortedKernelAdapter(DictEncoding).lift());
