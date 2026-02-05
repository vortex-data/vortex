// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::SliceReduce;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;

impl SliceReduce for VarBinVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        VarBinVTable::_slice(array, range).map(Some)
    }
}

impl VarBinVTable {
    pub fn _slice(array: &VarBinArray, range: Range<usize>) -> VortexResult<ArrayRef> {
        Ok(unsafe {
            VarBinArray::new_unchecked(
                array.offsets().slice(range.start..range.end + 1)?,
                array.bytes().clone(),
                array.dtype().clone(),
                array.validity()?.slice(range)?,
            )
            .into_array()
        })
    }
}
