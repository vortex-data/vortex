// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use crate::register_kernel;

impl IsConstantKernel for FixedSizeListVTable {
    fn is_constant(
        &self,
        array: &FixedSizeListArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        todo!()
    }
}

register_kernel!(IsConstantKernelAdapter(FixedSizeListVTable).lift());
