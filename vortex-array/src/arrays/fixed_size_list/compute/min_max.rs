// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::ListScalar;
use vortex_scalar::Scalar;

use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::register_kernel;

/// MinMax implementation for [`FixedSizeListArray`].
impl MinMaxKernel for FixedSizeListVTable {
    fn min_max(&self, array: &FixedSizeListArray) -> VortexResult<Option<MinMaxResult>> {
        let mut min: Option<Scalar> = None;
        let mut max: Option<Scalar> = None;
        for i in 0..array.len() {
            let scalar = array.scalar_at(i)?;
            if scalar.is_null() {
                continue;
            }
            let list_scalar = ListScalar::try_from(&scalar)?;
            if let Some(current_min) = &min {
                let current_min_list = ListScalar::try_from(current_min)?;
                if list_scalar < current_min_list {
                    min = Some(scalar.clone());
                }
            } else {
                min = Some(scalar.clone());
            }
            if let Some(current_max) = &max {
                let current_max_list = ListScalar::try_from(current_max)?;
                if list_scalar > current_max_list {
                    max = Some(scalar.clone());
                }
            } else {
                max = Some(scalar.clone());
            }
        }
        match (min, max) {
            (Some(min), Some(max)) => Ok(Some(MinMaxResult { min, max })),
            (None, None) => Ok(None),
            _ => unreachable!("min and max should be set together or both remain None"),
        }
    }
}

register_kernel!(MinMaxKernelAdapter(FixedSizeListVTable).lift());
