// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Slice;
use crate::arrays::SliceArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Slice {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let inner_range = array.slice_range();

        let combined_start = inner_range.start + range.start;
        let combined_end = inner_range.start + range.end;

        Ok(Some(
            SliceArray::new(array.child().clone(), combined_start..combined_end).into_array(),
        ))
    }
}
