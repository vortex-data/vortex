use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{ScalarAtFn, scalar_at};

impl ScalarAtFn<&ChunkedArray> for ChunkedEncoding {
    fn scalar_at(&self, array: &ChunkedArray, index: usize) -> VortexResult<Scalar> {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index);
        scalar_at(array.chunk(chunk_index)?, chunk_offset)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::{ChunkedArray, PrimitiveArray};
    use crate::compute::scalar_at;

    #[test]
    fn empty_children_both_sides() {
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
        assert_eq!(scalar_at(&array, 0).unwrap(), 1u64.into());
        assert_eq!(scalar_at(&array, 1).unwrap(), 2u64.into());
    }

    #[test]
    fn empty_children_trailing() {
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
        assert_eq!(scalar_at(&array, 0).unwrap(), 1u64.into());
        assert_eq!(scalar_at(&array, 1).unwrap(), 2u64.into());
        assert_eq!(scalar_at(&array, 2).unwrap(), 3u64.into());
        assert_eq!(scalar_at(&array, 3).unwrap(), 4u64.into());
    }

    #[test]
    fn empty_children_leading() {
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
        assert_eq!(scalar_at(&array, 0).unwrap(), 1u64.into());
        assert_eq!(scalar_at(&array, 1).unwrap(), 2u64.into());
        assert_eq!(scalar_at(&array, 2).unwrap(), 3u64.into());
        assert_eq!(scalar_at(&array, 3).unwrap(), 4u64.into());
    }
}
