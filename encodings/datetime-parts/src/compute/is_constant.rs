// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::IsConstantKernel;
use vortex_array::compute::IsConstantKernelAdapter;
use vortex_array::compute::IsConstantOpts;
use vortex_array::compute::is_constant_opts;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

impl IsConstantKernel for DateTimePartsVTable {
    fn is_constant(
        &self,
        array: &DateTimePartsArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let Some(days) = is_constant_opts(array.days().as_ref(), opts)? else {
            return Ok(None);
        };
        if !days {
            return Ok(Some(false));
        }

        let Some(seconds) = is_constant_opts(array.seconds().as_ref(), opts)? else {
            return Ok(None);
        };
        if !seconds {
            return Ok(Some(false));
        }

        let Some(subseconds) = is_constant_opts(array.subseconds().as_ref(), opts)? else {
            return Ok(None);
        };
        if !subseconds {
            return Ok(Some(false));
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(DateTimePartsVTable).lift());
