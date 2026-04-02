// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for VarBin {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        VarBin::_slice(array, range).map(Some)
    }
}

impl VarBin {
    pub fn _slice(array: ArrayView<'_, VarBin>, range: Range<usize>) -> VortexResult<ArrayRef> {
        Ok(unsafe {
            VarBinArray::new_unchecked_from_handle(
                array.offsets().slice(range.start..range.end + 1)?,
                array.bytes_handle().clone(),
                array.dtype().clone(),
                array.validity().slice(range)?,
            )
            .into_array()
        })
    }
}
