use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::unary::{scalar_at, ScalarAtFn};

impl ScalarAtFn<ChunkedArray> for ChunkedEncoding {
    fn scalar_at(&self, array: &ChunkedArray, index: usize) -> VortexResult<Scalar> {
        let (chunk_index, chunk_offset) = array.find_chunk_idx(index);
        scalar_at(&array.chunk(chunk_index)?, chunk_offset)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::{ChunkedArray, PrimitiveArray};
    use crate::compute::unary::scalar_at;
    use crate::IntoArrayData;

    #[test]
    fn empty_children_both_sides() {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(vec![1u64, 2]).into_array(),
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(scalar_at(array.as_ref(), 0).unwrap(), 1u64.into());
        assert_eq!(scalar_at(array.as_ref(), 1).unwrap(), 2u64.into());
    }

    #[test]
    fn empty_children_trailing() {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from(vec![1u64, 2]).into_array(),
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(vec![3u64, 4]).into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(scalar_at(array.as_ref(), 0).unwrap(), 1u64.into());
        assert_eq!(scalar_at(array.as_ref(), 1).unwrap(), 2u64.into());
        assert_eq!(scalar_at(array.as_ref(), 2).unwrap(), 3u64.into());
        assert_eq!(scalar_at(array.as_ref(), 3).unwrap(), 4u64.into());
    }

    #[test]
    fn empty_children_leading() {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(Vec::<u64>::new()).into_array(),
                PrimitiveArray::from(vec![1u64, 2]).into_array(),
                PrimitiveArray::from(vec![3u64, 4]).into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(scalar_at(array.as_ref(), 0).unwrap(), 1u64.into());
        assert_eq!(scalar_at(array.as_ref(), 1).unwrap(), 2u64.into());
        assert_eq!(scalar_at(array.as_ref(), 2).unwrap(), 3u64.into());
        assert_eq!(scalar_at(array.as_ref(), 3).unwrap(), 4u64.into());
    }
}
