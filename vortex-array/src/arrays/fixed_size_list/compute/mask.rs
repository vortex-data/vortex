// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for FixedSizeList {
    fn mask(
        array: ArrayView<'_, FixedSizeList>,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        Ok(Some(
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array.elements().clone(),
                    array.list_size(),
                    array.validity().and(Validity::Array(mask.clone()))?,
                    array.len(),
                )
            }
            .into_array(),
        ))
    }
}
