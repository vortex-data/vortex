// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::IsSortedKernel;
use vortex_array::compute::IsSortedKernelAdapter;
use vortex_array::compute::is_sorted;
use vortex_array::compute::is_strict_sorted;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndVTable;

impl IsSortedKernel for RunEndVTable {
    fn is_sorted(&self, array: &RunEndArray) -> VortexResult<Option<bool>> {
        is_sorted(array.values())
    }

    fn is_strict_sorted(&self, array: &RunEndArray) -> VortexResult<Option<bool>> {
        is_strict_sorted(array.to_canonical()?.as_ref())
    }
}

register_kernel!(IsSortedKernelAdapter(RunEndVTable).lift());
