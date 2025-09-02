// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_scalar::Scalar;

use crate::arrays::ChunkedVTable;
use crate::arrays::chunked::ChunkedArray;
use crate::vtable::OperationsVTable;
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<ChunkedVTable> for ChunkedVTable {
    fn slice(array: &ChunkedArray, range: Range<usize>) -> ArrayRef {
        assert!(
            !array.is_empty() || (range.start > 0 && range.end > 0),
            "Empty chunked array can't be sliced from {} to {}",
            range.start,
            range.end
        );

        if array.is_empty() {
            // SAFETY: empty chunked array trivially satisfies all validations
            unsafe {
                return ChunkedArray::new_unchecked(vec![], array.dtype().clone()).into_array();
            }
        }

        let (offset_chunk, offset_in_first_chunk) = array.find_chunk_idx(range.start);
        let (length_chunk, length_in_last_chunk) = array.find_chunk_idx(range.end);

        if length_chunk == offset_chunk {
            let chunk = array.chunk(offset_chunk);
            return chunk.slice(offset_in_first_chunk..length_in_last_chunk);
        }

        let mut chunks = (offset_chunk..length_chunk + 1)
            .map(|i| array.chunk(i).clone())
            .collect_vec();
        if let Some(c) = chunks.first_mut() {
            *c = c.slice(offset_in_first_chunk..c.len());
        }

        if length_in_last_chunk == 0 {
            chunks.pop();
        } else if let Some(c) = chunks.last_mut() {
            *c = c.slice(0..length_in_last_chunk);
        }

        // SAFETY: all chunks still have same DType
        unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array() }
    }

    fn scalar_at(array: &ChunkedArray, index: usize) -> Scalar {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index);
        array.chunk(chunk_index).scalar_at(chunk_offset)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, NativePType, Nullability, PType};

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::{ChunkedArray, ChunkedVTable, PrimitiveArray};
    use crate::canonical::ToCanonical;

    fn chunked_array() -> ChunkedArray {
        ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter([1u64, 2, 3]).into_array(),
                PrimitiveArray::from_iter([4u64, 5, 6]).into_array(),
                PrimitiveArray::from_iter([7u64, 8, 9]).into_array(),
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
                PrimitiveArray::from_iter([1u64, 2]).into_array(),
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
                PrimitiveArray::from_iter([1u64, 2]).into_array(),
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                PrimitiveArray::from_iter([3u64, 4]).into_array(),
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
                PrimitiveArray::from_iter([1u64, 2]).into_array(),
                PrimitiveArray::from_iter([3u64, 4]).into_array(),
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
