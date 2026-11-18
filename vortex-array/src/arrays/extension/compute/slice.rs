// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Extension {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let storage = array.storage_array().slice(range)?;

        // SAFETY: slice preserves the storage dtype and selected values from array.
        let result = unsafe { ExtensionArray::new_unchecked(array.ext_dtype().clone(), storage) };
        Ok(Some(result.into_array()))
    }
}
