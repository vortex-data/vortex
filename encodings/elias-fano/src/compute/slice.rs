// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::EliasFano;
use crate::array::EliasFanoArraySlotsExt;

impl SliceReduce for EliasFano {
    /// Slice by recording where the slice starts, leaving every buffer alone: the sample tables
    /// hold *absolute* bit positions and the low-bits child is packed in 1024-element blocks, so
    /// one rank offset covers both. See
    /// [`EliasFanoData::first_rank`](crate::EliasFanoData::first_rank).
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let data = array
            .data()
            .clone()
            .with_first_rank(array.first_rank() + range.start as u64);
        // SAFETY: this is a slice of an already-validated `EliasFanoArray`, so this is still valid.
        let sliced = unsafe { EliasFano::new_unchecked(data, array.lower().clone(), range.len()) };
        Ok(Some(sliced.into_array()))
    }
}
