// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl MaskKernel for StructVTable {
    fn mask(&self, array: &StructArray, filter_mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().mask(filter_mask);

        StructArray::try_new_with_dtype(
            array.unmasked_fields().clone(),
            array.struct_fields().clone(),
            array.len(),
            validity,
        )
        .map(|a| a.into_array())
    }
}
register_kernel!(MaskKernelAdapter(StructVTable).lift());
