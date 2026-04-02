// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for ListView {
    fn mask(array: ArrayView<'_, ListView>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    array.offsets().clone(),
                    array.sizes().clone(),
                    array.validity().and(Validity::Array(mask.clone()))?,
                )
                .with_zero_copy_to_list(array.is_zero_copy_to_list())
            }
            .into_array(),
        ))
    }
}
