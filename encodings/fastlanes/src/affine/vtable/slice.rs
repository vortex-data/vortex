// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::Affine;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;

impl SliceReduce for Affine {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // Keep the parameters of the chunks the slice overlaps, and record how far into the first
        // of them the slice starts, as FoR does.
        let start = usize::from(array.offset()) + range.start;
        let end = usize::from(array.offset()) + range.end;
        let chunks = start / FL_CHUNK_SIZE..end.div_ceil(FL_CHUNK_SIZE);
        Ok(Some(
            Affine::try_new(
                array.encoded().slice(range)?,
                array.references().slice(chunks.clone())?,
                array.scales().slice(chunks.clone())?,
                array.slopes().slice(chunks)?,
                u16::try_from(start % FL_CHUNK_SIZE)?,
                array.slope_shift(),
            )?
            .into_array(),
        ))
    }
}
