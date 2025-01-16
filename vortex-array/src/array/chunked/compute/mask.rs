use itertools::Itertools as _;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use super::filter::{chunk_filters, find_chunk_idx, ChunkFilter};
use crate::array::{ChunkedArray, ChunkedEncoding, ConstantArray};
use crate::compute::{mask, try_cast, FilterIter, FilterMask, MaskFn};
use crate::{ArrayDType, ArrayData, ArrayLen as _, IntoArrayData, IntoCanonical as _};

impl MaskFn<ChunkedArray> for ChunkedEncoding {
    fn mask(&self, array: &ChunkedArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let new_dtype = array.dtype().as_nullable();
        let new_chunks = match mask.iter() {
            FilterIter::Indices(_) => mask_indices(array, mask, &new_dtype),
            FilterIter::Slices(_) => mask_slices(array, mask, &new_dtype),
        }?;
        debug_assert_eq!(new_chunks.len(), array.nchunks());
        debug_assert_eq!(
            new_chunks.iter().map(|x| x.len()).sum::<usize>(),
            array.len()
        );
        ChunkedArray::try_new(new_chunks, new_dtype).map(IntoArrayData::into_array)
    }
}

fn mask_indices(
    array: &ChunkedArray,
    filter_mask: FilterMask,
    new_dtype: &DType,
) -> VortexResult<Vec<ArrayData>> {
    let mut new_chunks = Vec::with_capacity(array.nchunks());
    let mut current_chunk_id = 0;
    let mut chunk_indices = Vec::new();

    // Avoid find_chunk_idx and use our own to avoid the overhead.
    // The array should only be some thousands of values in the general case.
    let chunk_ends = array.chunk_offsets().into_canonical()?.into_primitive()?;
    let chunk_ends = chunk_ends.as_slice::<u64>();

    for &set_index in filter_mask.indices() {
        let (chunk_id, index) = find_chunk_idx(set_index, chunk_ends);
        if chunk_id != current_chunk_id {
            let chunk = array
                .chunk(current_chunk_id)
                .vortex_expect("find_chunk_idx must return valid chunk ID");
            let masked_chunk = mask(&chunk, FilterMask::from_indices(chunk.len(), chunk_indices))?;
            chunk_indices = Vec::new();
            new_chunks.push(masked_chunk);
            current_chunk_id += 1;

            // Advance the chunk forward, reset the chunk indices buffer.
            while current_chunk_id < chunk_id {
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
        let masked_chunk = mask(&chunk, FilterMask::from_indices(chunk.len(), chunk_indices))?;
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
    filter_mask: FilterMask,
    new_dtype: &DType,
) -> VortexResult<Vec<ArrayData>> {
    let chunked_filters = chunk_filters(array, &filter_mask)?;

    array
        .chunks()
        .zip_eq(chunked_filters.into_iter())
        .map(|(chunk, chunk_filter)| -> VortexResult<ArrayData> {
            Ok(match chunk_filter {
                ChunkFilter::All => {
                    // All => entire chunk is masked out
                    ConstantArray::new(Scalar::null(new_dtype.clone()), chunk.len()).into_array()
                }
                ChunkFilter::None => {
                    // None => preserve the entire chunk unmasked
                    chunk
                }
                // Slices => turn the slices into a boolean buffer.
                ChunkFilter::Slices(slices) => {
                    mask(&chunk, FilterMask::from_slices(chunk.len(), slices))?
                }
            })
        })
        .process_results(|iter| iter.collect::<Vec<_>>())
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::{ChunkedArray, PrimitiveArray};
    use crate::compute::test_harness::test_mask;
    use crate::IntoArrayData;

    #[test]
    fn test_mask_chunked_array() {
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![0u64, 1].into_array(),
                buffer![2_u64].into_array(),
                PrimitiveArray::empty::<u64>(dtype.nullability()).into_array(),
                buffer![3_u64, 4].into_array(),
            ],
            dtype,
        )
        .unwrap()
        .into_array();

        test_mask(chunked);
    }
}
