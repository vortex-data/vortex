// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

impl SliceReduce for FoR {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // Every chunk shares one reference, so the result needs no offset.
        if let Some(reference) = array.constant_reference() {
            return Ok(Some(
                FoR::try_new(array.encoded().slice(range)?, reference)?.into_array(),
            ));
        }

        // Keep the references of the chunks the slice overlaps, and record how far into the first
        // of them the slice starts.
        //
        // E.g. rows 1500..2500 of a 3000-row array with references [r0, r1, r2]:
        //   references = [r1, r2]  (the slice overlaps chunks 1 and 2)
        //   offset     = 476       (row 1500 is 476 rows into chunk 1)
        //
        // Adding the array's own offset first makes this work for already-sliced arrays too.
        let start = usize::from(array.offset()) + range.start;
        let end = usize::from(array.offset()) + range.end;
        let references = array
            .references()
            .slice(start / FL_CHUNK_SIZE..end.div_ceil(FL_CHUNK_SIZE))?;
        let offset = u16::try_from(start % FL_CHUNK_SIZE)?;
        Ok(Some(
            FoR::try_new_chunked(array.encoded().slice(range)?, references, offset)?.into_array(),
        ))
    }
}
