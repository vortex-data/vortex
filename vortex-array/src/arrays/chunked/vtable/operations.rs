// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Chunked;
use crate::scalar::Scalar;

impl OperationsVTable<Chunked> for Chunked {
    fn scalar_at(
        array: ArrayView<'_, Chunked>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index)?;
        array.chunk(chunk_index).scalar_at(chunk_offset)
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
