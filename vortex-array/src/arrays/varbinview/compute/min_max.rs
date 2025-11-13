// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewVTable, varbin_compute_min_max};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for VarBinViewVTable {
    fn min_max(&self, array: &VarBinViewArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(varbin_compute_min_max(array, array.dtype()))
    }
}

register_kernel!(MinMaxKernelAdapter(VarBinViewVTable).lift());
