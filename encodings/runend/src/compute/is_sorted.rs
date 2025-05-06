use vortex_array::compute::{IsSortedKernel, IsSortedKernelAdapter, is_sorted, is_strict_sorted};
use vortex_array::{Array, register_kernel};

use crate::{RunEndArray, RunEndEncoding};

impl IsSortedKernel for RunEndEncoding {
    fn is_sorted(&self, array: &RunEndArray) -> vortex_error::VortexResult<bool> {
        is_sorted(array.values())
    }

    fn is_strict_sorted(&self, array: &RunEndArray) -> vortex_error::VortexResult<bool> {
        is_strict_sorted(array.to_canonical()?.as_ref())
    }
}

register_kernel!(IsSortedKernelAdapter(RunEndEncoding).lift());
