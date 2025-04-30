use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use crate::register_kernel;

impl IsConstantKernel for VarBinEncoding {
    fn is_constant(
        &self,
        array: &VarBinArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        array.with_iterator(compute_is_constant).map(Some)
    }
}

register_kernel!(IsConstantKernelAdapter(VarBinEncoding).lift());

pub(super) fn compute_is_constant(iter: &mut dyn Iterator<Item = Option<&[u8]>>) -> bool {
    let Some(first_value) = iter.next() else {
        return false;
    };
    for v in iter {
        if v != first_value {
            return false;
        }
    }
    true
}
