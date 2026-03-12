// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::compute::IsConstantKernel;
use crate::compute::IsConstantKernelAdapter;
use crate::compute::IsConstantOpts;
use crate::compute::is_constant_opts;
use crate::register_kernel;

impl IsConstantKernel for Dict {
    fn is_constant(&self, array: &DictArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        if is_constant_opts(array.codes(), opts)? == Some(true) {
            return Ok(Some(true));
        }

        is_constant_opts(array.values(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(Dict).lift());
