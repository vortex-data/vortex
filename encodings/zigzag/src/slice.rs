// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::ZigZagArrayExt;
use crate::ZigZagVTable;

impl SliceReduce for ZigZagVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ZigZagVTable::new(array.encoded().slice(range)?).into_array(),
        ))
    }
}
