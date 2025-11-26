// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl MaskKernel for ListVTable {
    fn mask(&self, array: &ListArray, mask: &Mask) -> VortexResult<ArrayRef> {
        ListArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array.validity().mask(mask),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ListVTable).lift());
