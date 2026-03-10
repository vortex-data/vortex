// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::PatchedVTable;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for PatchedVTable {
    fn filter(array: &Self::Array, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        // Find the contiguous chunk range that the mask covers. We use this to slice the inner
        // components, then wrap the rest up with another FilterArray.
        //
        // This is helpful when we have a very selective filter that is clustered to a small
        // range.
        let (chunk_start, chunk_stop) = match mask.indices() {
            AllOr::All | AllOr::None => {
                // This is handled as the precondition to this method, see the FilterReduce
                // documentation.
                unreachable!("mask must be a MaskValues here")
            }
            AllOr::Some(indices) => {
                let first = indices[0];
                let last = indices[indices.len() - 1];

                (first / 1024, last.div_ceil(1024))
            }
        };

        // If all chunks already covered, there is nothing to do.
        if chunk_start == 0 && chunk_stop == array.n_chunks {
            return Ok(None);
        }

        let sliced = array.slice_chunks(chunk_start..chunk_stop)?;

        let slice_start = chunk_start * 1024;
        let slice_end = (chunk_start * 1024).min(array.len());
        let remainder = mask.slice(slice_start..slice_end);

        Ok(Some(
            FilterArray::new(sliced.into_array(), remainder).into_array(),
        ))
    }
}
