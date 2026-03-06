// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::fixed_size_list::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListVTable;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for FixedSizeListVTable {
    fn mask(array: &FixedSizeListArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        Ok(Some(
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array.elements().clone(),
                    array.list_size(),
                    array
                        .validity()
                        .clone()
                        .and(Validity::Array(mask.clone()))?,
                    array.len(),
                )
            }
            .into_array(),
        ))
    }
}
