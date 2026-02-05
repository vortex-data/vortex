// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_vector::binaryview::BinaryView;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::SliceReduce;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;

impl SliceReduce for VarBinViewVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            VarBinViewArray::new_handle(
                array
                    .views_handle()
                    .slice_typed::<BinaryView>(range.clone()),
                Arc::clone(array.buffers()),
                array.dtype().clone(),
                array.validity()?.slice(range)?,
            )
            .into_array(),
        ))
    }
}
