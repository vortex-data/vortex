// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::SequenceArray;
use crate::SequenceArrayExt;
use crate::SequenceVTable;

impl SliceReduce for SequenceVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: this is a slice of an already-validated `SequenceArray`, so this is still valid.
        Ok(Some(
            unsafe {
                SequenceArray::new_unchecked(
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
