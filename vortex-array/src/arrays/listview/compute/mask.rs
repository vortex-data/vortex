// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::listview::ListViewArray;
use crate::arrays::listview::ListViewVTable;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for ListViewVTable {
    fn mask(array: &ListViewArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    array.offsets().clone(),
                    array.sizes().clone(),
                    array
                        .validity()
                        .clone()
                        .and(Validity::Array(mask.clone()))?,
                )
                .with_zero_copy_to_list(array.is_zero_copy_to_list())
            }
            .into_array(),
        ))
    }
}
