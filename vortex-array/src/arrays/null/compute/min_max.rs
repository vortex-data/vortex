// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::null::NullArray;
use crate::arrays::null::NullVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for NullVTable {
    fn min_max(&self, _array: &NullArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(NullVTable).lift());
