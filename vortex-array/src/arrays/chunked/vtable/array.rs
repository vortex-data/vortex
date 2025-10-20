// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<ChunkedVTable> for ChunkedVTable {
    fn len(array: &ChunkedArray) -> usize {
        array.len
    }

    fn dtype(array: &ChunkedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ChunkedArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ChunkedArray, state: &mut H) {
        array.dtype.hash(state);
        array.len.hash(state);
        array.chunk_offsets.array_hash(state);
        for chunk in &array.chunks {
            chunk.array_hash(state);
        }
    }

    fn array_eq(array: &ChunkedArray, other: &ChunkedArray) -> bool {
        array.dtype == other.dtype
            && array.len == other.len
            && array.chunk_offsets.array_eq(&other.chunk_offsets)
            && array.chunks.len() == other.chunks.len()
            && array
                .chunks
                .iter()
                .zip(&other.chunks)
                .all(|(a, b)| a.array_eq(b))
    }
}
