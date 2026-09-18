// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::slice::SliceKernel;

impl SliceKernel for Chunked {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert!(
            !array.is_empty() || (range.start > 0 && range.end > 0),
            "Empty chunked array can't be sliced from {} to {}",
            range.start,
            range.end
        );

        if array.is_empty() {
            // SAFETY: empty chunked array trivially satisfies all validations
            unsafe {
                return Ok(Some(
                    ChunkedArray::new_unchecked([], array.dtype().clone()).into_array(),
                ));
            }
        }

        let (offset_chunk, offset_in_first_chunk) = array.find_chunk_idx(range.start)?;
        let (length_chunk, length_in_last_chunk) = array.find_chunk_idx(range.end)?;

        if length_chunk == offset_chunk {
            let chunk = array.chunk(offset_chunk);
            return Ok(Some(
                chunk.slice(offset_in_first_chunk..length_in_last_chunk)?,
            ));
        }

        let first = array.chunk(offset_chunk);
        let first = first.slice(offset_in_first_chunk..first.len())?;
        let middle = (offset_chunk + 1..length_chunk).map(|i| array.chunk(i).clone());
        let last = if length_in_last_chunk == 0 {
            None
        } else {
            Some(array.chunk(length_chunk).slice(0..length_in_last_chunk)?)
        };
        let chunks = std::iter::once(first).chain(middle).chain(last);

        // SAFETY: chunks are slices of the original valid chunks, preserving their dtype.
        // All chunks maintain the same dtype as the original array.
        Ok(Some(unsafe {
            ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array()
        }))
    }
}
