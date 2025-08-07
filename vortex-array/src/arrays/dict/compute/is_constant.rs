// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, is_constant_opts,
};
use crate::register_kernel;
use vortex_error::VortexResult;

use crate::arrays::{DictArray, DictVTable};

impl IsConstantKernel for DictVTable {
    fn is_constant(&self, array: &DictArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        if is_constant_opts(array.codes(), opts)? == Some(true) {
            return Ok(Some(true));
        }

        is_constant_opts(array.values(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(DictVTable).lift());
