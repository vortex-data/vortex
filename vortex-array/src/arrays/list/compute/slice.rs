// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for List {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ListArray::new(
                array.elements().clone(),
                array.offsets().slice(range.start..range.end + 1)?,
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}
