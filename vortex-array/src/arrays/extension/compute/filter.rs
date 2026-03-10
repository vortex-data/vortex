// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Extension {
    fn filter(array: &ExtensionArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            // SAFETY: The storage array is filtered from an already-valid extension array, which
            // preserves the storage dtype and does not change values.
            unsafe {
                ExtensionArray::new_unchecked(
                    array.ext_dtype().clone(),
                    array.storage_array().filter(mask.clone())?,
                )
            }
            .into_array(),
        ))
    }
}
