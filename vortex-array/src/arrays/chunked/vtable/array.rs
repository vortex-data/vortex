// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use crate::Precision;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ChunkedVTable> for ChunkedVTable {
    fn len(array: &ChunkedArray) -> usize {
        array.len
    }

    fn dtype(array: &ChunkedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ChunkedArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ChunkedArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.len.hash(state);
        array.chunk_offsets.as_ref().array_hash(state, precision);
        for chunk in &array.chunks {
            chunk.array_hash(state, precision);
        }
    }

    fn array_eq(array: &ChunkedArray, other: &ChunkedArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.len == other.len
            && array
                .chunk_offsets
                .as_ref()
                .array_eq(other.chunk_offsets.as_ref(), precision)
            && array.chunks.len() == other.chunks.len()
            && array
                .chunks
                .iter()
                .zip(&other.chunks)
                .all(|(a, b)| a.array_eq(b, precision))
    }
}
