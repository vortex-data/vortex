// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::DateTimeParts;
use crate::DateTimePartsData;

impl SliceReduce for DateTimeParts {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing all components preserves values
        Ok(Some(unsafe {
            DateTimePartsData::new_unchecked(
                array.dtype().clone(),
                array.days().slice(range.clone())?,
                array.seconds().slice(range.clone())?,
                array.subseconds().slice(range)?,
            )
            .into_array()
        }))
    }
}
