use crate::{DecimalBytePartsArray, DecimalBytePartsVTable};
use vortex_array::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts,
};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

impl IsConstantKernel for DecimalBytePartsVTable {
    fn is_constant(
        &self,
        array: &DecimalBytePartsArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        is_constant_opts(&array.msp, opts)
    }
}

register_kernel!(IsConstantKernelAdapter(DecimalBytePartsVTable).lift());
