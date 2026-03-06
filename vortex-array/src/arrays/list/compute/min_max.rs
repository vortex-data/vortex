// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::list::ListArray;
use crate::arrays::list::ListVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for ListVTable {
    fn min_max(&self, _array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // This would require finding the lexicographically minimum and maximum lists.
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(ListVTable).lift());
