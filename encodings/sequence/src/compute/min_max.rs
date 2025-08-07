// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use vortex_array::register_kernel;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::SequenceArray;
use crate::array::SequenceVTable;

impl MinMaxKernel for SequenceVTable {
    fn min_max(&self, array: &SequenceArray) -> VortexResult<Option<MinMaxResult>> {
        let base = array.base();
        let last = array.last();
        let (min, max) = if base < last {
            (base, last)
        } else {
            (last, base)
        };
        Ok(Some(MinMaxResult {
            min: Scalar::new(array.dtype().clone(), min.into()),
            max: Scalar::new(array.dtype().clone(), max.into()),
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(SequenceVTable).lift());
