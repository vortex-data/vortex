// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::SliceReduce;

impl SliceReduce for ExtensionVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ExtensionArray::new(array.ext_dtype().clone(), array.storage().slice(range)?)
                .into_array(),
        ))
    }
}
