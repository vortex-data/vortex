// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Bool {
    fn slice(array: ArrayView<'_, Bool>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            BoolArray::new(
                array.to_bit_buffer().slice(range.clone()),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}
