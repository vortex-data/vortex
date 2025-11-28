// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl MaskKernel for VarBinViewVTable {
    fn mask(&self, array: &VarBinViewArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // SAFETY: masking the validity does not affect the invariants
        unsafe {
            Ok(VarBinViewArray::new_unchecked(
                array.views().clone(),
                array.buffers().clone(),
                array.dtype().as_nullable(),
                array.validity().mask(mask)?,
            )
            .into_array())
        }
    }
}

register_kernel!(MaskKernelAdapter(VarBinViewVTable).lift());

#[cfg(test)]
mod tests {
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn take_mask_var_bin_view_array() {
        test_mask_conformance(
            VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).as_ref(),
        );

        test_mask_conformance(
            VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .as_ref(),
        );
    }
}
