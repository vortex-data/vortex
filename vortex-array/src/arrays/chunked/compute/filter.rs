// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::filter::FilterKernel;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

// This is modeled after the constant with the equivalent name in arrow-rs.
pub(crate) const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterKernel for Chunked {
    fn filter(
        array: ArrayView<'_, Chunked>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        // Based on filter selectivity, we take the values between a range of slices, or
        // we take individual indices.
        let chunks = match mask_values.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            MaskIter::Indices(indices) => filter_indices(array, indices.iter().copied()),
            MaskIter::Slices(slices) => filter_slices(array, slices.iter().copied()),
        }?;

        // SAFETY: Filter operation preserves the dtype of each chunk.
        // All filtered chunks maintain the same dtype as the original array.
        unsafe {
            Ok(Some(
                ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array(),
            ))
        }
    }
}

/// The filter to apply to each chunk.
///
/// When we rewrite a set of slices in a filter predicate into chunk addresses, we want to account
/// for the fact that some chunks will be wholly skipped.
#[derive(Clone)]
pub(crate) enum ChunkFilter {
    All,
    None,
    Slices(Vec<(usize, usize)>),
}

/// Filter the chunks using slice ranges.
fn filter_slices(
    array: ArrayView<'_, Chunked>,
    slices: impl Iterator<Item = (usize, usize)>,
) -> VortexResult<Vec<ArrayRef>> {
    let mut result = Vec::with_capacity(array.nchunks());

    let chunk_filters = chunk_filters(array, slices)?;

    // Now, apply the chunk filter to every slice.
    for (chunk, chunk_filter) in array.iter_chunks().zip(chunk_filters.into_iter()) {
        match chunk_filter {
            // All => preserve the entire chunk unfiltered.
            ChunkFilter::All => result.push(chunk.clone()),
            // None => whole chunk is filtered out, skip
            ChunkFilter::None => {}
            // Slices => turn the slices into a boolean buffer.
            ChunkFilter::Slices(slices) => {
                result.push(chunk.filter(Mask::from_slices(chunk.len(), slices))?);
            }
        }
    }

    Ok(result)
}

pub(crate) fn chunk_filters(
    array: ArrayView<'_, Chunked>,
    slices: impl Iterator<Item = (usize, usize)>,
) -> VortexResult<Vec<ChunkFilter>> {
    let chunk_offsets = array.chunk_offsets();

    let mut chunk_filters = vec![ChunkFilter::None; array.nchunks()];

    for (slice_start, slice_end) in slices {
        let (start_chunk, start_idx) = find_chunk_idx(slice_start, &chunk_offsets)?;
        // NOTE: we adjust slice end back by one, in case it ends on a chunk boundary, we do not
        // want to index into the unused chunk.
        let (end_chunk, end_idx) = find_chunk_idx(slice_end - 1, &chunk_offsets)?;
        // Adjust back to an exclusive range
        let end_idx = end_idx + 1;

        if start_chunk == end_chunk {
            // start == end means that the slice lies within a single chunk.
            match &mut chunk_filters[start_chunk] {
                f @ (ChunkFilter::All | ChunkFilter::None) => {
                    *f = ChunkFilter::Slices(vec![(start_idx, end_idx)]);
                }
                ChunkFilter::Slices(slices) => {
                    slices.push((start_idx, end_idx));
                }
            }
        } else {
            // start != end means that the range is split over at least two chunks:
            // start chunk: append a slice from (start_idx, start_chunk_end), i.e. whole chunk.
            // end chunk: append a slice from (0, end_idx).
            // chunks between start and end: append ChunkFilter::All.
            let start_chunk_len: usize =
                (chunk_offsets[start_chunk + 1] - chunk_offsets[start_chunk]).try_into()?;
            let start_slice = (start_idx, start_chunk_len);
            match &mut chunk_filters[start_chunk] {
                f @ (ChunkFilter::All | ChunkFilter::None) => {
                    *f = ChunkFilter::Slices(vec![start_slice])
                }
                ChunkFilter::Slices(slices) => slices.push(start_slice),
            }

            let end_slice = (0, end_idx);
            match &mut chunk_filters[end_chunk] {
                f @ (ChunkFilter::All | ChunkFilter::None) => {
                    *f = ChunkFilter::Slices(vec![end_slice]);
                }
                ChunkFilter::Slices(slices) => slices.push(end_slice),
            }

            for chunk in &mut chunk_filters[start_chunk + 1..end_chunk] {
                *chunk = ChunkFilter::All;
            }
        }
    }

    Ok(chunk_filters)
}

/// Filter the chunks using indices.
fn filter_indices(
    array: ArrayView<'_, Chunked>,
    indices: impl Iterator<Item = usize>,
) -> VortexResult<Vec<ArrayRef>> {
    let mut result = Vec::with_capacity(array.nchunks());
    let mut current_chunk_id = 0;
    let mut chunk_indices = BufferMut::with_capacity(array.nchunks());

    let chunk_offsets = array.chunk_offsets();

    for set_index in indices {
        let (chunk_id, index) = find_chunk_idx(set_index, &chunk_offsets)?;
        if chunk_id != current_chunk_id {
            // Push the chunk we've accumulated.
            if !chunk_indices.is_empty() {
                let chunk = array.chunk(current_chunk_id);
                let indices =
                    PrimitiveArray::new(chunk_indices.clone().freeze(), Validity::NonNullable);
                result.push(chunk.take(indices.into_array())?);
            }

            // Advance the chunk forward, reset the chunk indices buffer.
            current_chunk_id = chunk_id;
            chunk_indices.clear();
        }

        chunk_indices.push(index as u64);
    }

    if !chunk_indices.is_empty() {
        let chunk = array.chunk(current_chunk_id);
        let indices = PrimitiveArray::new(chunk_indices.clone().freeze(), Validity::NonNullable);
        let filtered_chunk = chunk.take(indices.into_array())?;
        result.push(filtered_chunk);
    }

    Ok(result)
}

/// Mirrors the find_chunk_idx method on ChunkedArray, but avoids all of the overhead
/// from scalars, dtypes, and metadata cloning.
pub(crate) fn find_chunk_idx(idx: usize, chunk_ends: &[u64]) -> VortexResult<(usize, usize)> {
    let chunk_id = chunk_ends
        .search_sorted(&(idx as u64), SearchSortedSide::Right)?
        .to_ends_index(chunk_ends.len())
        .saturating_sub(1);
    let chunk_begin: usize = chunk_ends[chunk_id]
        .try_into()
        .vortex_expect("chunk end must fit in usize");
    let chunk_offset = idx - chunk_begin;

    Ok((chunk_id, chunk_offset))
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::half::f16;

    #[test]
    fn filter_chunked_floats() {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![f16::from_f32(0.1463623)].into_array(),
                buffer![
                    f16::NAN,
                    f16::from_f32(0.24987793),
                    f16::from_f32(0.22497559),
                    f16::from_f32(0.22497559),
                    f16::from_f32(-36160.0),
                ]
                .into_array(),
                buffer![
                    f16::NAN,
                    f16::NAN,
                    f16::from_f32(0.22497559),
                    f16::from_f32(0.22497559),
                    f16::from_f32(3174.0),
                ]
                .into_array(),
            ],
            DType::Primitive(PType::F16, Nullability::NonNullable),
        )
        .unwrap();
        let mask = Mask::from_iter([
            true, false, false, true, true, true, true, true, true, true, true,
        ]);
        let filtered = chunked.filter(mask).unwrap();
        assert_eq!(filtered.len(), 9);
    }

    use rstest::rstest;

    #[rstest]
    #[case(ChunkedArray::try_new(
        vec![
            buffer![0u64, 1].into_array(),
            buffer![2_u64].into_array(),
            PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array(),
            buffer![3_u64, 4].into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(0u64), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(2u64)]).into_array(),
            PrimitiveArray::empty::<u64>(Nullability::Nullable).into_array(),
            PrimitiveArray::from_option_iter([None, Some(4u64)]).into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::Nullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        vec![
            buffer![1i32].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case(ChunkedArray::try_new(
        (0..10).map(|i| buffer![i as i64, i as i64 + 10, i as i64 + 20].into_array()).collect(),
        DType::Primitive(PType::I64, Nullability::NonNullable),
    ).unwrap())]
    fn test_filter_chunked_conformance(#[case] chunked: ChunkedArray) {
        test_filter_conformance(&chunked.into_array());
    }
}
