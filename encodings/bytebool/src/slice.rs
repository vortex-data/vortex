// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;

use crate::ByteBoolArray;
use crate::ByteBoolVTable;

impl SliceReduce for ByteBoolVTable {
    fn slice(array: &ByteBoolArray, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBoolArray::new(
                array.buffer().slice(range.clone()),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}
