// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{StructArray, StructVTable};
use crate::compute::{self, IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use crate::register_kernel;

impl IsConstantKernel for StructVTable {
    fn is_constant(
        &self,
        array: &StructArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let children = array.children();
        if children.is_empty() {
            return Ok(Some(true));
        }

        for child in children.iter() {
            match compute::is_constant_opts(child, opts)? {
                // Un-determined
                None => return Ok(None),
                Some(false) => return Ok(Some(false)),
                Some(true) => {}
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(StructVTable).lift());
