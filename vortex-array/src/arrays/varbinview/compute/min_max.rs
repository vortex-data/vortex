use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewEncoding, compute_min_max};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::{Array, register_kernel};

impl MinMaxKernel for VarBinViewEncoding {
    fn min_max(&self, array: &VarBinViewArray) -> VortexResult<Option<MinMaxResult>> {
        compute_min_max(array, array.dtype())
    }
}

register_kernel!(MinMaxKernelAdapter(VarBinViewEncoding).lift());
