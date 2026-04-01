// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::ByteBool;
use crate::ByteBoolData;

impl SliceReduce for ByteBool {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBoolData::new(
                array.buffer().slice(range.clone()),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}
