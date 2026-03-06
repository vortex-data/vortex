// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::list::ListArray;
use crate::arrays::list::ListVTable;
use crate::arrays::slice::SliceReduce;
use crate::vtable::ValidityHelper;

impl SliceReduce for ListVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
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
