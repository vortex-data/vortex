// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for VarBinViewVTable {
    fn mask(array: &VarBinViewArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        unsafe {
            Ok(Some(
                VarBinViewArray::new_handle_unchecked(
                    array.views_handle().clone(),
                    array.buffers().clone(),
                    array.dtype().as_nullable(),
                    array
                        .validity()
                        .clone()
                        .and(Validity::Array(mask.clone()))?,
                )
                .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn take_mask_var_bin_view_array() {
        test_mask_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).to_array(),
        );

        test_mask_conformance(
            &VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .to_array(),
        );
    }
}
