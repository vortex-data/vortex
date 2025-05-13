use vortex_buffer::BufferMut;
use vortex_dtype::PType;
use vortex_error::VortexResult;

use crate::arrays::ChunkedVTable;
use crate::arrays::chunked::ChunkedArray;
use crate::compute::{TakeKernel, TakeKernelAdapter, cast, take};
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for ChunkedVTable {
    fn take(&self, array: &ChunkedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = cast(indices, PType::U64.into())?.to_primitive()?;

        // While the chunk idx remains the same, accumulate a list of chunk indices.
        let mut chunks = Vec::new();
        let mut indices_in_chunk = BufferMut::<u64>::empty();
        let mut prev_chunk_idx = array
            .find_chunk_idx(indices.as_slice::<u64>()[0].try_into()?)
            .0;
        for idx in indices.as_slice::<u64>() {
            let idx = usize::try_from(*idx)?;
            let (chunk_idx, idx_in_chunk) = array.find_chunk_idx(idx);

            if chunk_idx != prev_chunk_idx {
                // Start a new chunk
                let indices_in_chunk_array = indices_in_chunk.clone().into_array();
                chunks.push(take(array.chunk(prev_chunk_idx)?, &indices_in_chunk_array)?);
                indices_in_chunk.clear();
            }

            indices_in_chunk.push(idx_in_chunk as u64);
            prev_chunk_idx = chunk_idx;
        }

        if !indices_in_chunk.is_empty() {
            let indices_in_chunk_array = indices_in_chunk.into_array();
            chunks.push(take(array.chunk(prev_chunk_idx)?, &indices_in_chunk_array)?);
        }

        Ok(ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array())
    }
}

register_kernel!(TakeKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::chunked::ChunkedArray;
    use crate::canonical::ToCanonical;
    use crate::compute::take;

    #[test]
    fn test_take() {
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = buffer![0u64, 0, 6, 4].into_array();

        let result = take(arr.as_ref(), indices.as_ref())
            .unwrap()
            .to_primitive()
            .unwrap();
        assert_eq!(result.as_slice::<i32>(), &[1, 1, 1, 2]);
    }
}
