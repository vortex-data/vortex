// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for BoolVTable {
    fn mask(&self, array: &BoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            BoolArray::new(array.boolean_buffer().clone(), array.validity().mask(mask)?)
                .into_array(),
        )
    }
}

register_kernel!(MaskKernelAdapter(BoolVTable).lift());

#[cfg(test)]
mod test {
    use crate::arrays::BoolArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn test_mask_bool_array() {
        test_mask_conformance(BoolArray::from_iter([true, false, true, true, false]).as_ref());

        // Test nullable bool array
        test_mask_conformance(
            BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]).as_ref(),
        );
    }
}
