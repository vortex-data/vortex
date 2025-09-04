// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for FixedSizeListVTable {
    fn mask(&self, array: &FixedSizeListArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // SAFETY: The only thing that changes here is the validity mask, which will have the same
        // length. So as long as the original array is valid, this is also valid.
        Ok(unsafe {
            FixedSizeListArray::new_unchecked(
                array.elements().clone(),
                array.list_size(),
                array.validity().mask(mask),
                array.len(),
            )
        }
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(FixedSizeListVTable).lift());
