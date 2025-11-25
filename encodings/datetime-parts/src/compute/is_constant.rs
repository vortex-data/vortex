// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::IsConstantKernel;
use vortex_array::compute::IsConstantKernelAdapter;
use vortex_array::compute::IsConstantOpts;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

impl IsConstantKernel for DateTimePartsVTable {
    fn is_constant(
        &self,
        array: &DateTimePartsArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        Ok(Some(
            array.days().is_constant()
                && array.seconds().is_constant()
                && array.subseconds().is_constant(),
        ))
    }
}

register_kernel!(IsConstantKernelAdapter(DateTimePartsVTable).lift());
