// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::Array;
use crate::arrays::{
    ChunkedArray,
    ChunkedVTable,
};
use crate::vtable::ValidityVTable;

impl ValidityVTable<ChunkedVTable> for ChunkedVTable {
    fn is_valid(array: &ChunkedArray, index: usize) -> bool {
        if !array.dtype.is_nullable() {
            return true;
        }
        let (chunk, offset_in_chunk) = array.find_chunk_idx(index);
        array.chunk(chunk).is_valid(offset_in_chunk)
    }

    fn all_valid(array: &ChunkedArray) -> bool {
        if !array.dtype().is_nullable() {
            return true;
        }
        for chunk in array.non_empty_chunks() {
            if !chunk.all_valid() {
                return false;
            }
        }
        true
    }

    fn all_invalid(array: &ChunkedArray) -> bool {
        if !array.dtype().is_nullable() {
            return false;
        }
        for chunk in array.non_empty_chunks() {
            if !chunk.all_invalid() {
                return false;
            }
        }
        true
    }

    fn validity_mask(array: &ChunkedArray) -> Mask {
        array.chunks().iter().map(|a| a.validity_mask()).collect()
    }
}
