// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for VarBinView {
    fn mask(array: ArrayView<'_, VarBinView>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        unsafe {
            Ok(Some(
                VarBinViewArray::new_handle_unchecked(
                    array.views_handle().clone(),
                    Arc::clone(array.data_buffers()),
                    array.dtype().as_nullable(),
                    array.validity()?.and(Validity::Array(mask.clone()))?,
                )
                .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::IntoArray;
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn take_mask_var_bin_view_array() {
        test_mask_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).into_array(),
        );

        test_mask_conformance(
            &VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .into_array(),
        );
    }
}
