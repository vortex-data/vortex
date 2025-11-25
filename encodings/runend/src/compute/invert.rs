// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::compute::InvertKernel;
use vortex_array::compute::InvertKernelAdapter;
use vortex_array::compute::invert;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndVTable;

impl InvertKernel for RunEndVTable {
    fn invert(&self, array: &RunEndArray) -> VortexResult<ArrayRef> {
        // SAFETY: ends are preserved
        unsafe {
            Ok(RunEndArray::new_unchecked(
                array.ends().clone(),
                invert(array.values())?,
                array.len(),
                array.offset(),
            )
            .into_array())
        }
    }
}

register_kernel!(InvertKernelAdapter(RunEndVTable).lift());
