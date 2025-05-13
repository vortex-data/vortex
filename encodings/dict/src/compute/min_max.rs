use vortex_array::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult, min_max, take};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{DictArray, DictVTable};

impl MinMaxKernel for DictVTable {
    fn min_max(&self, array: &DictArray) -> VortexResult<Option<MinMaxResult>> {
        min_max(&take(array.values(), array.codes())?)
    }
}

register_kernel!(MinMaxKernelAdapter(DictVTable).lift());
