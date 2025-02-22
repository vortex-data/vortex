use vortex_error::{vortex_bail, VortexResult};

use crate::arrays::chunked::ChunkedArray;
use crate::arrays::ChunkedEncoding;
use crate::compute::{slice, SliceFn};
use crate::{Array, ArrayRef, IntoArray};

impl SliceFn<&ChunkedArray> for ChunkedEncoding {
    fn slice(&self, array: &ChunkedArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let (offset_chunk, offset_in_first_chunk) = array.find_chunk_idx(start);
        let (length_chunk, length_in_last_chunk) = array.find_chunk_idx(stop);

        if array.is_empty() && (start != 0 || stop != 0) {
            vortex_bail!(ComputeError: "Empty chunked array can't be sliced from {start} to {stop}");
        } else if array.is_empty() {
            return Ok(ChunkedArray::new_unchecked(vec![], array.dtype().clone()).into_array());
        }

        if length_chunk == offset_chunk {
            let chunk = array.chunk(offset_chunk)?;
            return Ok(ChunkedArray::new_unchecked(
                vec![slice(chunk, offset_in_first_chunk, length_in_last_chunk)?],
                array.dtype().clone(),
            )
            .into_array());
        }

        let mut chunks = (offset_chunk..length_chunk + 1)
            .map(|i| array.chunk(i).cloned())
            .collect::<VortexResult<Vec<_>>>()?;
        if let Some(c) = chunks.first_mut() {
            *c = slice(c, offset_in_first_chunk, c.len())?;
        }

        if length_in_last_chunk == 0 {
            chunks.pop();
        } else if let Some(c) = chunks.last_mut() {
            *c = slice(c, 0, length_in_last_chunk)?;
        }

        Ok(ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, NativePType, Nullability, PType};

    use crate::arrays::{ChunkedArray, PrimitiveArray};
    use crate::canonical::ToCanonical;
    use crate::compute::slice;
    use crate::{Array, ArrayExt, ArrayRef, IntoArray};

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
        arr.as_::<ChunkedArray>()
            .chunks()
            .iter()
            .map(|a| a.to_primitive().unwrap())
            .for_each(|a| values.extend_from_slice(a.as_slice::<T>()));
        assert_eq!(values, slice);
    }

    #[test]
    fn slice_middle() {
        assert_equal_slices(&slice(&chunked_array(), 2, 5).unwrap(), &[3u64, 4, 5])
    }

    #[test]
    fn slice_begin() {
        assert_equal_slices(&slice(&chunked_array(), 1, 3).unwrap(), &[2u64, 3]);
    }

    #[test]
    fn slice_aligned() {
        assert_equal_slices(&slice(&chunked_array(), 3, 6).unwrap(), &[4u64, 5, 6]);
    }

    #[test]
    fn slice_many_aligned() {
        assert_equal_slices(
            &slice(&chunked_array(), 0, 6).unwrap(),
            &[1u64, 2, 3, 4, 5, 6],
        );
    }

    #[test]
    fn slice_end() {
        assert_equal_slices(&slice(&chunked_array(), 7, 8).unwrap(), &[8u64]);
    }

    #[test]
    fn slice_exactly_end() {
        assert_equal_slices(&slice(&chunked_array(), 6, 9).unwrap(), &[7u64, 8, 9]);
    }

    #[test]
    fn slice_empty() {
        let chunked = ChunkedArray::try_new(vec![], PType::U32.into()).unwrap();
        let sliced = slice(&chunked, 0, 0).unwrap();

        assert!(sliced.is_empty());
    }
}
