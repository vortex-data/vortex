// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::varbin_compute_min_max;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for VarBinView {
    fn min_max(&self, array: &VarBinViewArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(varbin_compute_min_max(array, array.dtype()))
    }
}

register_kernel!(MinMaxKernelAdapter(VarBinView).lift());
