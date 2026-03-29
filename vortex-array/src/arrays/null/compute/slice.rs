// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::slice::SliceReduce;
use crate::vtable::Array;

impl SliceReduce for Null {
    fn slice(_array: &Array<Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(NullArray::new(range.len()).into_array()))
    }
}
