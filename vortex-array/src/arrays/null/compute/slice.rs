// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Null {
    fn slice(_array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(NullArray::new(range.len()).into_array()))
    }
}
