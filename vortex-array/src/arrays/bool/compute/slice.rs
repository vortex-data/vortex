// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::SliceReduce;
use crate::vtable::ValidityHelper;

impl SliceReduce for BoolVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            BoolArray::new(
                array.to_bit_buffer().slice(range.clone()),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}
