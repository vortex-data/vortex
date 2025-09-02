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
        FixedSizeListArray::try_new(
            array.elements().clone(),
            array.list_size(),
            array.validity().mask(mask),
            array.len(),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(MaskKernelAdapter(FixedSizeListVTable).lift());
