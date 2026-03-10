// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for ExtensionVTable {
    fn filter(array: &ExtensionArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ExtensionArray::new(
                array.ext_dtype().clone(),
                array.storage_array().filter(mask.clone())?,
            )
            .into_array(),
        ))
    }
}
