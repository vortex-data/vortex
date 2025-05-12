use vortex_array::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts,
};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{DictArray, DictVTable};

impl IsConstantKernel for DictVTable {
    fn is_constant(&self, array: &DictArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant_opts(array.codes(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(DictVTable).lift());
