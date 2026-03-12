// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::IsConstantKernel;
use vortex_array::compute::IsConstantKernelAdapter;
use vortex_array::compute::IsConstantOpts;
use vortex_array::compute::is_constant_opts;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::FoR;
use crate::FoRArray;

impl IsConstantKernel for FoR {
    fn is_constant(&self, array: &FoRArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        is_constant_opts(array.encoded(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(FoR).lift());
