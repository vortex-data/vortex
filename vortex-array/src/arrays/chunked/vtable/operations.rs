// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ChunkedVTable> for ChunkedVTable {
    fn scalar_at(array: &ChunkedArray, index: usize) -> Scalar {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index);
        array.chunk(chunk_index).scalar_at(chunk_offset)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::NativePType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ChunkedVTable;
    use crate::canonical::ToCanonical;

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

    fn assert_equal_slices<T: NativePType>(arr: &dyn Array, slice: &[T]) {
        let mut values = Vec::with_capacity(arr.len());
        if let Some(arr) = arr.as_opt::<ChunkedVTable>() {
            arr.chunks()
                .iter()
                .map(|a| a.to_primitive())
                .for_each(|a| values.extend_from_slice(a.as_slice::<T>()));
        } else {
            values.extend_from_slice(arr.to_primitive().as_slice::<T>());
        }
        assert_eq!(values, slice);
    }

    #[test]
    fn slice_middle() {
        assert_equal_slices(&chunked_array().slice(2..5), &[3u64, 4, 5])
    }

    #[test]
    fn slice_begin() {
        assert_equal_slices(&chunked_array().slice(1..3), &[2u64, 3]);
    }

    #[test]
    fn slice_aligned() {
        assert_equal_slices(&chunked_array().slice(3..6), &[4u64, 5, 6]);
    }

    #[test]
    fn slice_many_aligned() {
        assert_equal_slices(&chunked_array().slice(0..6), &[1u64, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn slice_end() {
        assert_equal_slices(&chunked_array().slice(7..8), &[8u64]);
    }

    #[test]
    fn slice_exactly_end() {
        assert_equal_slices(&chunked_array().slice(6..9), &[7u64, 8, 9]);
    }

    #[test]
    fn slice_empty() {
        let chunked = ChunkedArray::try_new(vec![], PType::U32.into()).unwrap();
        let sliced = chunked.slice(0..0);

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
        assert_eq!(array.scalar_at(0), 1u64.into());
        assert_eq!(array.scalar_at(1), 2u64.into());
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
        assert_eq!(array.scalar_at(0), 1u64.into());
        assert_eq!(array.scalar_at(1), 2u64.into());
        assert_eq!(array.scalar_at(2), 3u64.into());
        assert_eq!(array.scalar_at(3), 4u64.into());
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
        assert_eq!(array.scalar_at(0), 1u64.into());
        assert_eq!(array.scalar_at(1), 2u64.into());
        assert_eq!(array.scalar_at(2), 3u64.into());
        assert_eq!(array.scalar_at(3), 4u64.into());
    }
}
