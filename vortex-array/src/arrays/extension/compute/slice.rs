// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Extension {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            // SAFETY: The storage array is sliced from an already-valid extension array, which
            // preserves the storage dtype and does not change values.
            unsafe {
                ExtensionArray::new_unchecked(
                    array.ext_dtype().clone(),
                    array.storage_array().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}
