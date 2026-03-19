// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::slice::SliceReduce;
use crate::arrays::varbinview::BinaryView;

impl SliceReduce for VarBinView {
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

impl FilterReduce for VarBinView {
    fn filter(array: &VarBinViewArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let ranges: Vec<Range<usize>> = mask
            .slices()
            .unwrap_or_else(|| unreachable!(), || unreachable!())
            .iter()
            .map(|&(s, e)| s..e)
            .collect();
        Ok(Some(
            VarBinViewArray::new_handle(
                array.views_handle().filter_typed::<BinaryView>(&ranges)?,
                Arc::clone(array.buffers()),
                array.dtype().clone(),
                array.validity()?.filter(mask)?,
            )
            .into_array(),
        ))
    }
}
