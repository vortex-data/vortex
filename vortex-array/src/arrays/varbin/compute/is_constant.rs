// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::arrays::{VarBinArray, VarBinVTable};
use crate::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use crate::register_kernel;

impl IsConstantKernel for VarBinVTable {
    fn is_constant(
        &self,
        array: &VarBinArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        if opts.is_negligible_cost() {
            return Ok(None);
        }

        Ok(Some(compute_is_constant(array.iter())))
    }
}

register_kernel!(IsConstantKernelAdapter(VarBinVTable).lift());

pub(super) fn compute_is_constant(mut iter: impl Iterator<Item = Option<ByteBuffer>>) -> bool {
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
