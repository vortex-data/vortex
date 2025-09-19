// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for ListViewVTable {
    fn min_max(&self, _array: &ListViewArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(connor)[ListView]: Implement min_max for ListView.
        // This would require finding the lexicographically minimum and maximum lists.
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(ListViewVTable).lift());
