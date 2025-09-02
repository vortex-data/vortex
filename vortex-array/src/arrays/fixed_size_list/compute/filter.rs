// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::FixedSizeListVTable;
use crate::compute::{FilterKernel, FilterKernelAdapter};
use crate::{ArrayRef, register_kernel};

impl FilterKernel for FixedSizeListVTable {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        todo!()
    }
}

register_kernel!(FilterKernelAdapter(FixedSizeListVTable).lift());
