// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::BlockedFoR;
use crate::blocked_for::array::BLOCK_SIZE;
use crate::blocked_for::array::BlockedFoRArrayExt;
use crate::blocked_for::array::BlockedFoRArraySlotsExt;

impl SliceReduce for BlockedFoR {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let encoded = array.encoded().slice(range.clone())?;

        // An empty slice keeps no references at all, so it must also reset the offset: a
        // non-zero offset would otherwise imply a (non-existent) partial first block.
        if range.is_empty() {
            let references = array.references().slice(0..0)?;
            return Ok(Some(
                BlockedFoR::try_new(encoded, references, 0)?.into_array(),
            ));
        }

        let start = range.start + array.offset() as usize;
        let end = range.end + array.offset() as usize;
        let references = array
            .references()
            .slice(start / BLOCK_SIZE..(end - 1) / BLOCK_SIZE + 1)?;

        Ok(Some(
            BlockedFoR::try_new(encoded, references, (start % BLOCK_SIZE) as u16)?.into_array(),
        ))
    }
}
