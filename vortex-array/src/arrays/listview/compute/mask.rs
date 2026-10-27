// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewData;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for ListView {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, ListView>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.and(Validity::Array(mask.clone()))?;
        let slots = ListViewData::make_slots(
            array.elements(),
            array.offsets(),
            array.sizes(),
            &validity,
            array.len(),
        );
        let parts = ArrayParts::new(
            ListView,
            array.dtype().as_nullable(),
            array.len(),
            array.data().clone(),
            slots,
        );

        // SAFETY: elements, offsets, sizes, and their metadata are unchanged. Masking only removes
        // valid rows, so the existing zero-copy-to-list guarantee still holds.
        Ok(Some(
            unsafe { ListViewArray::from_parts_unchecked(parts) }.into_array(),
        ))
    }
}
