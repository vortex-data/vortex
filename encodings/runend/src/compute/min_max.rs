use vortex_array::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult, min_max};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl MinMaxKernel for RunEndEncoding {
    fn min_max(&self, array: &RunEndArray) -> VortexResult<Option<MinMaxResult>> {
        min_max(array.values())
    }
}

register_kernel!(MinMaxKernelAdapter(RunEndEncoding).lift());
