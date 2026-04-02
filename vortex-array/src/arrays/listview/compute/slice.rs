// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for ListView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    array.offsets().slice(range.clone())?,
                    array.sizes().slice(range.clone())?,
                    array.validity().slice(range)?,
                )
                .with_zero_copy_to_list(array.is_zero_copy_to_list())
            }
            .into_array(),
        ))
    }
}
