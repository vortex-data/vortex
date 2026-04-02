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
        Ok(Some(
            VarBinViewArray::new_handle(
                array
                    .views_handle()
                    .slice_typed::<BinaryView>(range.clone()),
                Arc::clone(array.data_buffers()),
                array.dtype().clone(),
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }
}
