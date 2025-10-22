// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{NullArray, NullVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for NullVTable {
    fn min_max(&self, _array: &NullArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(NullVTable).lift());
