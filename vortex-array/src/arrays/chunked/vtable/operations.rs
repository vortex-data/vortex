// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

impl OperationsVTable<Chunked> for Chunked {
    fn scalar_at(
        array: ArrayView<'_, Chunked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index)?;
        array.chunk(chunk_index).execute_scalar(chunk_offset, ctx)
    }

    /// Route to the chunk containing `index`, then recurse via the dispatch
    /// so the session caches at every level.
    fn point_scalar_at(
        array: ArrayView<'_, Chunked>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index)?;
        let chunk = array.chunk(chunk_index).clone();
        d.scalar_at(&chunk, chunk_offset)
    }

    /// `search_sorted` on a cross-chunk-monotonic Chunked array: identify the
    /// candidate chunk by inspecting each chunk's last element (one scalar_at
    /// per chunk), then descend into that one chunk and translate the result
    /// back to logical (whole-array) coordinates.
    ///
    /// Precondition (caller's responsibility): the chunks taken together are
    /// sorted, i.e. each chunk's max ≤ the next chunk's min. The default
    /// `OperationsVTable::point_search_sorted` (generic binary search) handles
    /// arbitrary Chunked shapes correctly; this override is a strict speedup
    /// only when the precondition holds.
    fn point_search_sorted(
        array: ArrayView<'_, Chunked>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let nchunks = array.nchunks();
        let offsets = array.chunk_offsets();
        let total_len = array.as_ref().len();

        // Find the first chunk whose last element is ≥ the target (for Left)
        // or > the target (for Right). Chunk count is typically small enough
        // that linear scanning is fine; one scalar_at per chunk on its last
        // index gives the bound.
        for chunk_idx in 0..nchunks {
            let chunk = array.chunk(chunk_idx).clone();
            let chunk_len = chunk.len();
            if chunk_len == 0 {
                continue;
            }
            let last = d.scalar_at(&chunk, chunk_len - 1)?;
            let last_cmp = last.partial_cmp(value);
            let could_contain = match (last_cmp, side) {
                // Chunk's last is < value → target is in a later chunk.
                (Some(Ordering::Less), _) => false,
                // Chunk's last == value with Right side: the boundary is past
                // the rightmost equal element, which might be at chunk end.
                // The next chunk (if any) starts with values > target, so the
                // boundary is here.
                _ => true,
            };
            if !could_contain {
                continue;
            }
            let local = d.search_sorted(&chunk, value, side)?;
            let chunk_start = offsets[chunk_idx];
            return Ok(match local {
                SearchResult::Found(i) => SearchResult::Found(i + chunk_start),
                SearchResult::NotFound(i) => SearchResult::NotFound(i + chunk_start),
            });
        }

        // Value greater than every chunk's last element.
        Ok(SearchResult::NotFound(total_len))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    fn chunked_array() -> ChunkedArray {
        ChunkedArray::try_new(
            vec![
                buffer![1u64, 2, 3].into_array(),
                buffer![4u64, 5, 6].into_array(),
                buffer![7u64, 8, 9].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap()
    }

    #[rstest]
    #[case::middle(2..5, &[3u64, 4, 5])]
    #[case::begin(1..3, &[2u64, 3])]
    #[case::aligned(3..6, &[4u64, 5, 6])]
    #[case::many_aligned(0..6, &[1u64, 2, 3, 4, 5, 6])]
    #[case::end(7..8, &[8u64])]
    #[case::exactly_end(6..9, &[7u64, 8, 9])]
    fn slice(#[case] range: Range<usize>, #[case] expected: &[u64]) {
        assert_arrays_eq!(
            chunked_array().slice(range).unwrap(),
            PrimitiveArray::from_iter(expected.iter().copied())
        );
    }

    #[test]
    fn slice_empty() {
        let chunked = ChunkedArray::try_new(vec![], PType::U32.into()).unwrap();
        let sliced = chunked.slice(0..0).unwrap();

        assert!(sliced.is_empty());
    }

    #[test]
    fn scalar_at_empty_children_both_sides() {
        let array = ChunkedArray::try_new(
            vec![
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                buffer![1u64, 2].into_array(),
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2]));
    }

    #[test]
    fn scalar_at_empty_children_trailing() {
        let array = ChunkedArray::try_new(
            vec![
                buffer![1u64, 2].into_array(),
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                buffer![3u64, 4].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2, 3, 4]));
    }

    #[test]
    fn scalar_at_empty_children_leading() {
        let array = ChunkedArray::try_new(
            vec![
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                buffer![1u64, 2].into_array(),
                buffer![3u64, 4].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2, 3, 4]));
    }
}
