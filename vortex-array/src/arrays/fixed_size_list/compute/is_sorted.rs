// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

/// IsSorted implementation for [`FixedSizeListArray`].
impl IsSortedKernel for FixedSizeListVTable {
    fn is_sorted(&self, _array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        // This would require comparing lists lexicographically.
        Ok(None)
    }

    fn is_strict_sorted(&self, _array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        // This would require comparing lists lexicographically without duplicates.
        Ok(None)
    }
}

register_kernel!(IsSortedKernelAdapter(FixedSizeListVTable).lift());
