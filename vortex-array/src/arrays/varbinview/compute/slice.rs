// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::slice::SliceReduce;
use crate::arrays::varbinview::BinaryView;

impl SliceReduce for VarBinView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let views = array
            .views_handle()
            .slice_typed::<BinaryView>(range.clone());
        let data_buffers = Arc::clone(array.data_buffers());
        let dtype = array.dtype().clone();
        let validity = array.validity()?.slice(range)?;

        // Safety:
        // range is validated within bounds, and is shared between all children.
        let array = unsafe {
            VarBinViewArray::new_handle_unchecked(views, data_buffers, dtype, validity).into_array()
        };

        Ok(Some(array))
    }
}
