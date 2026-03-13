// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

/// MinMax implementation for [`FixedSizeListArray`].
impl MinMaxKernel for FixedSizeList {
    fn min_max(&self, _array: &FixedSizeListArray) -> VortexResult<Option<MinMaxResult>> {
        // This would require finding the lexicographically minimum and maximum lists.
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(FixedSizeList).lift());
