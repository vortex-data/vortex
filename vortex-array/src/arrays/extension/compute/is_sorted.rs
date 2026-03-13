// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::compute::{self};
use crate::register_kernel;

impl IsSortedKernel for Extension {
    fn is_sorted(&self, array: &ExtensionArray) -> VortexResult<Option<bool>> {
        compute::is_sorted(array.storage_array())
    }

    fn is_strict_sorted(&self, array: &ExtensionArray) -> VortexResult<Option<bool>> {
        compute::is_strict_sorted(array.storage_array())
    }
}

register_kernel!(IsSortedKernelAdapter(Extension).lift());
