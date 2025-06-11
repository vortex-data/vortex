use vortex_array::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts,
};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::RunEndVTable;

impl IsConstantKernel for RunEndVTable {
    fn is_constant(
        &self,
        array: &Self::Array,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        // If there are known to be me 0 len runs then we can check if constant on the values.
        if is_constant_opts(array.values(), opts)? == Some(true) {
            return Ok(Some(true));
        }
        Ok(None)
    }
}

register_kernel!(IsConstantKernelAdapter(RunEndVTable).lift());
