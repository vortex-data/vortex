// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Extension {
    fn filter(array: ArrayView<'_, Extension>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let storage = array.storage_array().filter(mask.clone())?;

        // SAFETY: filter preserves the storage dtype and selected values from array.
        let result = unsafe { ExtensionArray::new_unchecked(array.ext_dtype().clone(), storage) };
        Ok(Some(result.into_array()))
    }
}
