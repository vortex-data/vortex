// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::Sequence;

impl SliceReduce for Sequence {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: this is a slice of an already-validated `SequenceArray`, so this is still valid.
        Ok(Some(
            unsafe {
                Sequence::new_unchecked(
                    array.index_value(range.start),
                    array.multiplier(),
                    array.calculation_ptype(),
                    array.dtype().as_ptype(),
                    array.dtype().nullability(),
                    range.len(),
                )
            }
            .into_array(),
        ))
    }
}
