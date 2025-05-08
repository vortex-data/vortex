use vortex_array::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts,
};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{FoRArray, FoREncoding};

impl IsConstantKernel for FoREncoding {
    fn is_constant(&self, array: &FoRArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant_opts(array.encoded(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(FoREncoding).lift());
