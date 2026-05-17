// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use super::take::sum_sizes_if_cheap;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for ListView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let new_sizes = array.sizes().slice(range.clone())?;
        let bound = sum_sizes_if_cheap(&new_sizes);
        let sliced = unsafe {
            ListViewArray::new_unchecked(
                array.elements().clone(),
                array.offsets().slice(range.clone())?,
                new_sizes,
                array.validity()?.slice(range)?,
            )
            .with_zero_copy_to_list(array.is_zero_copy_to_list())
        }
        .with_reachable_elements_bound(bound);
        Ok(Some(sliced.into_array()))
    }
}
