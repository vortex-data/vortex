use itertools::Itertools as _;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_mask::{AllOr, Mask, MaskIter};
use vortex_scalar::Scalar;

use super::filter::{ChunkFilter, chunk_filters, find_chunk_idx};
use crate::arrays::chunked::compute::filter::FILTER_SLICES_SELECTIVITY_THRESHOLD;
use crate::arrays::{ChunkedArray, ChunkedEncoding, ConstantArray};
use crate::compute::{MaskFn, mask, try_cast};
use crate::{Array, ArrayRef};

impl MaskFn<&ChunkedArray> for ChunkedEncoding {
    fn mask(&self, array: &ChunkedArray, mask: Mask) -> VortexResult<ArrayRef> {
        let new_dtype = array.dtype().as_nullable();
        let new_chunks = match mask.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            AllOr::All => unreachable!("handled in top-level mask"),
            AllOr::None => unreachable!("handled in top-level mask"),
            AllOr::Some(MaskIter::Indices(indices)) => mask_indices(array, indices, &new_dtype),
            AllOr::Some(MaskIter::Slices(slices)) => {
                mask_slices(array, slices.iter().cloned(), &new_dtype)
            }
        }?;
        debug_assert_eq!(new_chunks.len(), array.nchunks());
        debug_assert_eq!(
            new_chunks.iter().map(|x| x.len()).sum::<usize>(),
            array.len()
        );
        ChunkedArray::try_new(new_chunks, new_dtype).map(|c| c.into_array())
    }
}

fn mask_indices(
    array: &ChunkedArray,
    indices: &[usize],
    new_dtype: &DType,
) -> VortexResult<Vec<ArrayRef>> {
    let mut new_chunks = Vec::with_capacity(array.nchunks());
    let mut current_chunk_id = 0;
    let mut chunk_indices = Vec::new();

    let chunk_offsets = array.chunk_offsets();

    for &set_index in indices {
        let (chunk_id, index) = find_chunk_idx(set_index, chunk_offsets);
        if chunk_id != current_chunk_id {
            let chunk = array
                .chunk(current_chunk_id)
                .vortex_expect("find_chunk_idx must return valid chunk ID");
            let masked_chunk = mask(chunk, Mask::from_indices(chunk.len(), chunk_indices))?;
            // Advance the chunk forward, reset the chunk indices buffer.
            chunk_indices = Vec::new();
            new_chunks.push(masked_chunk);
            current_chunk_id += 1;

            while current_chunk_id < chunk_id {
                // Chunks that are not affected by the mask, must still be casted to the correct dtype.
                let chunk = array
                    .chunk(current_chunk_id)
                    .vortex_expect("find_chunk_idx must return valid chunk ID");
                new_chunks.push(try_cast(chunk, new_dtype)?);
                current_chunk_id += 1;
            }
        }

        chunk_indices.push(index);
    }

    if !chunk_indices.is_empty() {
        let chunk = array
            .chunk(current_chunk_id)
            .vortex_expect("find_chunk_idx must return valid chunk ID");
        let masked_chunk = mask(chunk, Mask::from_indices(chunk.len(), chunk_indices))?;
        new_chunks.push(masked_chunk);
        current_chunk_id += 1;
    }

    while current_chunk_id < array.nchunks() {
        let chunk = array
            .chunk(current_chunk_id)
            .vortex_expect("find_chunk_idx must return valid chunk ID");
        new_chunks.push(try_cast(chunk, new_dtype)?);
        current_chunk_id += 1;
    }

    Ok(new_chunks)
}

fn mask_slices(
    array: &ChunkedArray,
    slices: impl Iterator<Item = (usize, usize)>,
    new_dtype: &DType,
) -> VortexResult<Vec<ArrayRef>> {
    let chunked_filters = chunk_filters(array, slices)?;

    array
        .chunks()
        .iter()
        .zip_eq(chunked_filters)
        .map(|(chunk, chunk_filter)| -> VortexResult<ArrayRef> {
            Ok(match chunk_filter {
                ChunkFilter::All => {
                    // entire chunk is masked out
                    ConstantArray::new(Scalar::null(new_dtype.clone()), chunk.len()).into_array()
                }
                ChunkFilter::None => {
                    // entire chunk is not affected by mask
                    chunk.clone()
                }
                ChunkFilter::Slices(slices) => {
                    // Slices of indices that must be set to null
                    mask(chunk, Mask::from_slices(chunk.len(), slices))?
                }
            })
        })
        .process_results(|iter| iter.collect::<Vec<_>>())
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::{ChunkedArray, PrimitiveArray};
    use crate::compute::test_harness::test_mask;

    #[test]
    fn test_mask_chunked_array() {
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![0u64, 1].into_array(),
                buffer![2_u64].into_array(),
                PrimitiveArray::empty::<u64>(dtype.nullability()).to_array(),
                buffer![3_u64, 4].into_array(),
            ],
            dtype,
        )
        .unwrap();

        test_mask(&chunked);
    }
}
