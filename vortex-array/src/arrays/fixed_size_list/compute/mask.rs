// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

/// Mask implementation for [`FixedSizeListArray`].
///
/// Applies a validity mask to the array without modifying the underlying element data.
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
