// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::validity::Validity;
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

    fn validity(array: &ChunkedArray) -> VortexResult<Validity> {
        Ok(Validity::Array(
            unsafe {
                ChunkedArray::new_unchecked(
                    array
                        .chunks()
                        .iter()
                        .map(|chunk| chunk.validity().map(|v| v.to_array(chunk.len())))
                        .try_collect()?,
                    DType::Bool(Nullability::NonNullable),
                )
            }
            .into_array(),
        ))
    }

    fn validity_mask(array: &ChunkedArray) -> Mask {
        array.chunks().iter().map(|a| a.validity_mask()).collect()
    }
}
