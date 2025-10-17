// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{NullArray, NullVTable};
use crate::compute::{FilterKernel, FilterKernelAdapter};
use crate::{ArrayRef, IntoArray, register_kernel};

impl FilterKernel for NullVTable {
    fn filter(&self, _array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(mask.true_count()).into_array())
    }
}

register_kernel!(FilterKernelAdapter(NullVTable).lift());
