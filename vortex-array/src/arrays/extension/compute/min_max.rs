// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for ExtensionVTable {
    fn min_max(&self, array: &ExtensionArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(
            compute::min_max(array.storage())?.map(|MinMaxResult { min, max }| MinMaxResult {
                min: Scalar::extension(array.ext_dtype().clone(), min),
                max: Scalar::extension(array.ext_dtype().clone(), max),
            }),
        )
    }
}

register_kernel!(MinMaxKernelAdapter(ExtensionVTable).lift());
