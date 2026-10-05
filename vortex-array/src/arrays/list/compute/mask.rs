// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::list::ListArraySlotsExt;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for List {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, List>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: elements and offsets are unchanged, and masking only removes valid rows.
        Ok(Some(unsafe {
            ListArray::new_unchecked(
                array.elements().clone(),
                array.offsets().clone(),
                array.validity()?.and(Validity::Array(mask.clone()))?,
            )
            .into_array()
        }))
    }
}
