// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use crate::register_kernel;

impl IsConstantKernel for ExtensionVTable {
    fn is_constant(
        &self,
        array: &ExtensionArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        compute::is_constant_opts(array.storage(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(ExtensionVTable).lift());
