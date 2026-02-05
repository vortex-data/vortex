// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

impl SliceReduce for DateTimePartsVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing all components preserves values
        Ok(Some(unsafe {
            DateTimePartsArray::new_unchecked(
                array.dtype().clone(),
                array.days().slice(range.clone())?,
                array.seconds().slice(range.clone())?,
                array.subseconds().slice(range)?,
            )
            .into_array()
        }))
    }
}
