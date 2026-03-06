// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::fixed_size_list::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

/// MinMax implementation for [`FixedSizeListArray`].
impl MinMaxKernel for FixedSizeListVTable {
    fn min_max(&self, _array: &FixedSizeListArray) -> VortexResult<Option<MinMaxResult>> {
        // This would require finding the lexicographically minimum and maximum lists.
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(FixedSizeListVTable).lift());
