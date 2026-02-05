// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_error::VortexResult;

use crate::PcoVTable;

impl SliceReduce for PcoVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(array._slice(range.start, range.end).into_array()))
    }
}
