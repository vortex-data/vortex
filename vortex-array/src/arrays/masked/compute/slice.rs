// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Masked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let child = array.child().slice(range.clone())?;
        let validity = array.validity()?.slice(range)?;

        // SAFETY: slicing a null-free child keeps it null-free, and the child and validity are
        // sliced by the same range, so their lengths match.
        Ok(Some(
            unsafe { MaskedArray::new_unchecked(child, validity) }.into_array(),
        ))
    }
}
