// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

impl MinMaxKernel for Struct {
    fn min_max(&self, _array: &StructArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement struct min max
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(Struct).lift());
