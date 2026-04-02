// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::Sequence;
use crate::SequenceData;

impl SliceReduce for Sequence {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: this is a slice of an already-validated `SequenceArray`, so this is still valid.
        Ok(Some(
            unsafe {
                SequenceData::new_unchecked(
                    array.index_value(range.start),
                    array.multiplier(),
                    array.ptype(),
                    array.dtype().nullability(),
                    range.len(),
                )
            }
            .into_array(),
        ))
    }
}
