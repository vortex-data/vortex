// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for FixedSizeList {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let new_len = range.len();
        let list_size = array.list_size() as usize;

        // SAFETY: Slicing preserves FixedSizeListArray invariants
        Ok(Some(
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array
                        .elements()
                        .slice(range.start * list_size..range.end * list_size)?,
                    array.list_size(),
                    array.validity().slice(range)?,
                    new_len,
                )
            }
            .into_array(),
        ))
    }
}
