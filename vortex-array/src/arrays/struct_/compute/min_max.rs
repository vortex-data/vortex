// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{StructArray, StructVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for StructVTable {
    fn min_max(&self, _array: &StructArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement struct min max
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(StructVTable).lift());
