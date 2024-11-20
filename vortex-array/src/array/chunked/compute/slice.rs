use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::array::ChunkedEncoding;
use crate::compute::{slice, SliceFn};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl SliceFn<ChunkedArray> for ChunkedEncoding {
    fn slice(&self, array: &ChunkedArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let (offset_chunk, offset_in_first_chunk) = array.find_chunk_idx(start);
        let (length_chunk, length_in_last_chunk) = array.find_chunk_idx(stop);

        if length_chunk == offset_chunk {
            let chunk = array.chunk(offset_chunk)?;
            return Ok(ChunkedArray::try_new(
                vec![slice(&chunk, offset_in_first_chunk, length_in_last_chunk)?],
                array.dtype().clone(),
            )?
            .into_array());
        }

        let mut chunks = (offset_chunk..length_chunk + 1)
            .map(|i| array.chunk(i))
            .collect::<VortexResult<Vec<_>>>()?;
        if let Some(c) = chunks.first_mut() {
            *c = slice(&*c, offset_in_first_chunk, c.len())?;
        }

        if length_in_last_chunk == 0 {
            chunks.pop();
        } else if let Some(c) = chunks.last_mut() {
            *c = slice(&*c, 0, length_in_last_chunk)?;
        }

        ChunkedArray::try_new(chunks, array.dtype().clone()).map(|a| a.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, NativePType, Nullability, PType};

    use crate::array::ChunkedArray;
    use crate::compute::slice;
    use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

    fn chunked_array() -> ChunkedArray {
        ChunkedArray::try_new(
            vec![
                vec![1u64, 2, 3].into_array(),
                vec![4u64, 5, 6].into_array(),
                vec![7u64, 8, 9].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap()
    }

    fn assert_equal_slices<T: NativePType>(arr: ArrayData, slice: &[T]) {
        let mut values = Vec::with_capacity(arr.len());
        ChunkedArray::try_from(arr)
            .unwrap()
            .chunks()
            .map(|a| a.into_primitive().unwrap())
            .for_each(|a| values.extend_from_slice(a.maybe_null_slice::<T>()));
        assert_eq!(values, slice);
    }

    #[test]
    pub fn slice_middle() {
        assert_equal_slices(
            slice(chunked_array().as_ref(), 2, 5).unwrap(),
            &[3u64, 4, 5],
        )
    }

    #[test]
    pub fn slice_begin() {
        assert_equal_slices(slice(chunked_array().as_ref(), 1, 3).unwrap(), &[2u64, 3]);
    }

    #[test]
    pub fn slice_aligned() {
        assert_equal_slices(
            slice(chunked_array().as_ref(), 3, 6).unwrap(),
            &[4u64, 5, 6],
        );
    }

    #[test]
    pub fn slice_many_aligned() {
        assert_equal_slices(
            slice(chunked_array().as_ref(), 0, 6).unwrap(),
            &[1u64, 2, 3, 4, 5, 6],
        );
    }

    #[test]
    pub fn slice_end() {
        assert_equal_slices(slice(chunked_array().as_ref(), 7, 8).unwrap(), &[8u64]);
    }

    #[test]
    pub fn slice_exactly_end() {
        assert_equal_slices(
            slice(chunked_array().as_ref(), 6, 9).unwrap(),
            &[7u64, 8, 9],
        );
    }
}
