// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::register_kernel;

impl IsConstantKernel for VarBin {
    fn is_constant(
        &self,
        array: &VarBinArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        if opts.is_negligible_cost() {
            return Ok(None);
        }
        Ok(Some(array.with_iterator(compute_is_constant)))
    }
}

register_kernel!(IsConstantKernelAdapter(VarBin).lift());

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
