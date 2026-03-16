// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::IsConstantKernel;
use vortex_array::compute::IsConstantKernelAdapter;
use vortex_array::compute::IsConstantOpts;
use vortex_array::compute::is_constant_opts;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;

impl IsConstantKernel for DecimalByteParts {
    fn is_constant(
        &self,
        array: &DecimalBytePartsArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        is_constant_opts(&array.msp, opts)
    }
}

register_kernel!(IsConstantKernelAdapter(DecimalByteParts).lift());
