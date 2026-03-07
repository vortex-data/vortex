// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::MinMaxKernel;
use vortex_array::compute::MinMaxKernelAdapter;
use vortex_array::compute::MinMaxResult;
use vortex_array::compute::min_max;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndArrayExt;
use crate::RunEndVTable;

impl MinMaxKernel for RunEndVTable {
    fn min_max(&self, array: &RunEndArray) -> VortexResult<Option<MinMaxResult>> {
        min_max(array.values())
    }
}

register_kernel!(MinMaxKernelAdapter(RunEndVTable).lift());
