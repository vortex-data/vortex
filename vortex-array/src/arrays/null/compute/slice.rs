// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::arrays::SliceReduce;

impl SliceReduce for NullVTable {
    fn slice(_array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(NullArray::new(range.len()).into_array()))
    }
}
