// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for ListViewVTable {
    fn mask(&self, array: &ListViewArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // SAFETY: Since we are only masking the validity and everything else comes from an already
        // valid `ListViewArray`, all of the invariants are still upheld.
        Ok(unsafe {
            ListViewArray::new_unchecked(
                array.elements().clone(),
                array.offsets().clone(),
                array.sizes().clone(),
                array.validity().mask(mask),
                array.is_zero_copy_to_list(),
            )
        }
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(ListViewVTable).lift());
