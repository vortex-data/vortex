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
        ListViewArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array.sizes().clone(),
            array.validity().mask(mask),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ListViewVTable).lift());
