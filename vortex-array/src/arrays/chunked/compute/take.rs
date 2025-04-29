use itertools::Itertools;
use vortex_buffer::BufferMut;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::ChunkedEncoding;
use crate::arrays::chunked::ChunkedArray;
use crate::compute::{
    SearchSortedSide, TakeFn, cast, scalar_at, search_sorted_usize, slice, sub_scalar, take,
};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

impl TakeFn<&ChunkedArray> for ChunkedEncoding {
    fn take(&self, array: &ChunkedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // Fast path for strict sorted indices.
        if indices
            .statistics()
            .compute_is_strict_sorted()
            .unwrap_or(false)
        {
            if array.len() == indices.len() {
                return Ok(array.to_array().into_array());
            }

            return take_strict_sorted(array, indices);
        }

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

/// When the indices are non-null and strict-sorted, we can do better
fn take_strict_sorted(chunked: &ChunkedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
    let mut indices_by_chunk = vec![None; chunked.nchunks()];

    // Track our position in the indices array
    let mut pos = 0;
    while pos < indices.len() {
        // Locate the chunk index for the current index
        let idx = usize::try_from(&scalar_at(indices, pos)?)?;
        let (chunk_idx, _idx_in_chunk) = chunked.find_chunk_idx(idx);

        // Find the end of this chunk, and locate that position in the indices array.
        let chunk_begin = usize::try_from(chunked.chunk_offsets()[chunk_idx])?;
        let chunk_end = usize::try_from(chunked.chunk_offsets()[chunk_idx + 1])?;
        let chunk_end_pos =
            search_sorted_usize(indices, chunk_end, SearchSortedSide::Left)?.to_index();

        // Now we can say the slice of indices belonging to this chunk is [pos, chunk_end_pos)
        let chunk_indices = slice(indices, pos, chunk_end_pos)?;

        // Adjust the indices so they're relative to the chunk
        // Note. Indices might not have a dtype big enough to fit chunk_begin after cast,
        // if it does cast the scalar otherwise upcast the indices.
        let chunk_indices = if chunk_begin
            < PType::try_from(chunk_indices.dtype())?
                .max_value_as_u64()
                .try_into()?
        {
            sub_scalar(
                &chunk_indices,
                Scalar::from(chunk_begin).cast(chunk_indices.dtype())?,
            )?
        } else {
            // Note. this try_cast (memory copy) is unnecessary, could instead upcast in the subtract fn.
            //  and avoid an extra
            let u64_chunk_indices = cast(&chunk_indices, PType::U64.into())?;
            sub_scalar(&u64_chunk_indices, chunk_begin.into())?
        };

        indices_by_chunk[chunk_idx] = Some(chunk_indices);

        pos = chunk_end_pos;
    }

    // Now we can take the chunks
    let chunks = indices_by_chunk
        .into_iter()
        .enumerate()
        .filter_map(|(chunk_idx, indices)| indices.map(|i| (chunk_idx, i)))
        .map(|(chunk_idx, chunk_indices)| take(chunked.chunk(chunk_idx)?, &chunk_indices))
        .try_collect()?;

    Ok(ChunkedArray::try_new(chunks, chunked.dtype().clone())?.into_array())
}

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

        let result = take(&arr, &indices).unwrap().to_primitive().unwrap();
        assert_eq!(result.as_slice::<i32>(), &[1, 1, 1, 2]);
    }
}
