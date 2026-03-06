// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::compute::{self};
use crate::register_kernel;

impl IsSortedKernel for ExtensionVTable {
    fn is_sorted(&self, array: &ExtensionArray) -> VortexResult<Option<bool>> {
        compute::is_sorted(array.storage())
    }

    fn is_strict_sorted(&self, array: &ExtensionArray) -> VortexResult<Option<bool>> {
        compute::is_strict_sorted(array.storage())
    }
}

register_kernel!(IsSortedKernelAdapter(ExtensionVTable).lift());
