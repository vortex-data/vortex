// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_error::VortexResult;

use crate::FoRArray;
use crate::FoRVTable;

impl SliceReduce for FoRVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: Just slicing encoded data does not affect FOR.
        Ok(Some(unsafe {
            FoRArray::new_unchecked(
                array.encoded().slice(range)?,
                array.reference_scalar().clone(),
            )
            .into_array()
        }))
    }
}
