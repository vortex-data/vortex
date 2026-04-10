// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::slice::SliceReduce;
use crate::arrays::varbinview::BinaryView;

impl SliceReduce for VarBinView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            VarBinViewArray::new_handle(
                array
                    .views_handle()
                    .slice_typed::<BinaryView>(range.clone()),
                Arc::clone(array.data_buffers()),
                array.dtype().clone(),
                array.validity()?.slice(range)?,
            )
            .into_array(),
        ))
    }
}

impl FilterReduce for VarBinView {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        // For host buffers, only reduce if the filter is selective enough.
        // VarBinView is more expensive to filter (views + data buffers) so use
        // a lower threshold than Primitive.
        if array.views_handle().is_on_host() && mask.true_count() * 3 > mask.len() {
            return Ok(None);
        }
        Ok(Some(
            VarBinViewArray::new_handle(
                array.views_handle().filter(mask, size_of::<BinaryView>())?,
                Arc::clone(array.data_buffers()),
                array.dtype().clone(),
                array.validity()?.filter(mask)?,
            )
            .into_array(),
        ))
    }
}
