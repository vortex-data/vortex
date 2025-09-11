// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

// TODO(connor): Right now this does nothing.
/// IsSorted implementation for [`FixedSizeListArray`].
impl IsSortedKernel for FixedSizeListVTable {
    fn is_sorted(&self, _array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }

    fn is_strict_sorted(&self, _array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }
}

register_kernel!(IsSortedKernelAdapter(FixedSizeListVTable).lift());
