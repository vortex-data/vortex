// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;

impl MinMaxKernel for ConstantVTable {
    fn min_max(&self, array: &ConstantArray) -> VortexResult<Option<MinMaxResult>> {
        let scalar = array.scalar();
        if scalar.is_null() {
            return Ok(None);
        }
        let non_nullable_dtype = scalar.dtype().as_nonnullable();
        Ok(Some(MinMaxResult {
            min: scalar.cast(&non_nullable_dtype)?,
            max: scalar.cast(&non_nullable_dtype)?,
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(ConstantVTable).lift());
