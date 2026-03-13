// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::compute::{self};
use crate::register_kernel;

impl IsConstantKernel for Extension {
    fn is_constant(
        &self,
        array: &ExtensionArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        compute::is_constant_opts(array.storage_array(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(Extension).lift());
