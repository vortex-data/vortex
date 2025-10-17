// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};

impl FilterKernel for ExtensionVTable {
    fn filter(&self, array: &ExtensionArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            compute::filter(array.storage(), mask)?,
        )
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(ExtensionVTable).lift());
