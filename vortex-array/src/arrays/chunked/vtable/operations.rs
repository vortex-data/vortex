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

    fn point_kernels() -> Option<&'static crate::point_fn::PointKernels<Chunked>> {
        Some(&POINT_KERNELS)
    }
}

const POINT_KERNELS: crate::point_fn::PointKernels<Chunked> =
    crate::point_fn::PointKernels::empty()
        .with_scalar_at(crate::point_fn::PointKernels::lift_scalar_at(
            &ChunkedScalarAtKernel,
        ))
        .with_search_sorted(crate::point_fn::PointKernels::lift_search_sorted(
            &ChunkedSearchSortedKernel,
        ));

/// Route to the chunk containing `index`, then recurse via the dispatch so
/// the session caches at every level.
struct ChunkedScalarAtKernel;

impl crate::point_fn::ScalarAtKernel<Chunked> for ChunkedScalarAtKernel {
    fn execute(
        view: ArrayView<'_, Chunked>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        let (chunk_index, chunk_offset) = view.find_chunk_idx(index)?;
        let chunk = view.chunk(chunk_index).clone();
        d.scalar_at(&chunk, chunk_offset)
    }
}

/// `search_sorted` on a cross-chunk-monotonic Chunked array: identify the
/// candidate chunk by inspecting each chunk's last element (one scalar_at
/// per chunk), then descend into that one chunk and translate the result
/// back to logical (whole-array) coordinates.
///
/// Precondition (caller's responsibility): the chunks taken together are
/// sorted, i.e. each chunk's max ≤ the next chunk's min. The default
/// generic binary search handles arbitrary Chunked shapes correctly; this
/// override is a strict speedup only when the precondition holds.
struct ChunkedSearchSortedKernel;

impl crate::point_fn::SearchSortedKernel<Chunked> for ChunkedSearchSortedKernel {
    fn execute(
        view: ArrayView<'_, Chunked>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        let nchunks = view.nchunks();
        let offsets = view.chunk_offsets();
        let total_len = view.as_ref().len();

        for chunk_idx in 0..nchunks {
            let chunk = view.chunk(chunk_idx).clone();
            let chunk_len = chunk.len();
            if chunk_len == 0 {
                continue;
            }
            let last = d.scalar_at(&chunk, chunk_len - 1)?;
            // Chunk's last < target → target is in a later chunk; skip.
            if matches!(last.partial_cmp(value), Some(Ordering::Less)) {
                let _ = side;
                continue;
            }
            let local = d.search_sorted(&chunk, value, side)?;
            let chunk_start = offsets[chunk_idx];
            return Ok(match local {
                SearchResult::Found(i) => SearchResult::Found(i + chunk_start),
                SearchResult::NotFound(i) => SearchResult::NotFound(i + chunk_start),
            });
        }

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
