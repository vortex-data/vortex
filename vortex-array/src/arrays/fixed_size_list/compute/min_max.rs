// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

// TODO(connor): Right now this does nothing.
/// MinMax implementation for [`FixedSizeListArray`].
impl MinMaxKernel for FixedSizeListVTable {
    fn min_max(&self, _array: &FixedSizeListArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(FixedSizeListVTable).lift());
