// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::FoR;
use crate::FoRData;

impl SliceReduce for FoR {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: Just slicing encoded data does not affect FOR.
        Ok(Some(unsafe {
            FoRData::new_unchecked(
                array.encoded().slice(range)?,
                array.reference_scalar().clone(),
            )
            .into_array()
        }))
    }
}
