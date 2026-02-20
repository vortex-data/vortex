// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::MinMaxKernel;
use vortex_array::compute::MinMaxKernelAdapter;
use vortex_array::compute::MinMaxResult;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::register_kernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

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
            min: Scalar::primitive_value(min, array.ptype(), NonNullable),
            max: Scalar::primitive_value(max, array.ptype(), NonNullable),
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(SequenceVTable).lift());
