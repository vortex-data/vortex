// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ChunkedVTable> for ChunkedVTable {
    fn is_valid(array: &ChunkedArray, index: usize) -> VortexResult<bool> {
        if !array.dtype.is_nullable() {
            return Ok(true);
        }
        let (chunk, offset_in_chunk) = array.find_chunk_idx(index);
        array.chunk(chunk).is_valid(offset_in_chunk)
    }

    fn all_valid(array: &ChunkedArray) -> VortexResult<bool> {
        if !array.dtype().is_nullable() {
            return Ok(true);
        }
        for chunk in array.non_empty_chunks() {
            if !chunk.all_valid()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn all_invalid(array: &ChunkedArray) -> VortexResult<bool> {
        if !array.dtype().is_nullable() {
            return Ok(false);
        }
        for chunk in array.non_empty_chunks() {
            if !chunk.all_invalid()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn validity_mask(array: &ChunkedArray) -> VortexResult<Mask> {
        array
            .chunks()
            .iter()
            .map(|a| a.validity_mask())
            .collect::<Result<Mask, _>>()
    }
}
