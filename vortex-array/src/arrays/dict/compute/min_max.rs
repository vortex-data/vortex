// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult, min_max, take};
use crate::register_kernel;
use vortex_error::VortexResult;

use crate::arrays::{DictArray, DictVTable};

impl MinMaxKernel for DictVTable {
    fn min_max(&self, array: &DictArray) -> VortexResult<Option<MinMaxResult>> {
        min_max(&take(array.values(), array.codes())?)
    }
}

register_kernel!(MinMaxKernelAdapter(DictVTable).lift());
