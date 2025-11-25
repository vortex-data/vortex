// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

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
            )
            .with_zero_copy_to_list(array.is_zero_copy_to_list())
        }
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(ListViewVTable).lift());
