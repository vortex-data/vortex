// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::list::ListArray;
use crate::arrays::list::ListVTable;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::register_kernel;

impl IsSortedKernel for ListVTable {
    fn is_sorted(&self, _array: &ListArray) -> VortexResult<Option<bool>> {
        // This would require comparing lists lexicographically.
        Ok(None)
    }

    fn is_strict_sorted(&self, _array: &ListArray) -> VortexResult<Option<bool>> {
        // This would require comparing lists lexicographically without duplicates.
        Ok(None)
    }
}

register_kernel!(IsSortedKernelAdapter(ListVTable).lift());
