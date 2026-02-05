// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::arrays::SliceReduce;
use vortex_error::VortexResult;

use crate::SequenceArray;
use crate::SequenceVTable;

impl SliceReduce for SequenceVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            SequenceArray::unchecked_new(
                array.index_value(range.start),
                array.multiplier(),
                array.ptype(),
                array.dtype().nullability(),
                range.len(),
            )
            .to_array(),
        ))
    }
}
