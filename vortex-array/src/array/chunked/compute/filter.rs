use arrow_buffer::BooleanBufferBuilder;
use vortex_error::{VortexExpect, VortexResult};

use crate::array::{ChunkedArray, ChunkedEncoding, PrimitiveArray};
use crate::compute::{
    filter, take, FilterFn, FilterMask, SearchSorted, SearchSortedSide, TakeOptions,
};
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoCanonical};

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterFn<ChunkedArray> for ChunkedEncoding {
    fn filter(&self, array: &ChunkedArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let selected = mask.true_count();

        // Based on filter selectivity, we take the values between a range of slices, or
        // we take individual indices.
        let selectivity = selected as f64 / array.len() as f64;
        let chunks = if selectivity > FILTER_SLICES_SELECTIVITY_THRESHOLD {
            filter_slices(array, mask)
        } else {
            filter_indices(array, mask)
        };

        Ok(ChunkedArray::try_new(chunks?, array.dtype().clone())?.into_array())
    }
}

/// The filter to apply to each chunk.
///
/// When we rewrite a set of slices in a filter predicate into chunk addresses, we want to account
/// for the fact that some chunks will be wholly skipped.
#[derive(Clone)]
enum ChunkFilter {
    All,
    None,
    Slices(Vec<(usize, usize)>),
}

/// Given a sequence of slices that indicate ranges of set values, returns a boolean array
/// representing the same thing.
fn slices_to_mask(slices: &[(usize, usize)], len: usize) -> FilterMask {
    let mut buffer = BooleanBufferBuilder::new(len);

    let mut pos = 0;
    for (slice_start, slice_end) in slices.iter().copied() {
        // write however many trailing `false` between the end of the previous slice and the
        // start of this one.
        let n_leading_false = slice_start - pos;
        buffer.append_n(n_leading_false, false);
        buffer.append_n(slice_end - slice_start, true);
        pos = slice_end;
    }

    // Pad the end of the buffer with false, if necessary.
    let n_trailing_false = len - pos;
    buffer.append_n(n_trailing_false, false);

    FilterMask::from(buffer.finish())
}

/// Filter the chunks using slice ranges.
#[allow(deprecated)]
fn filter_slices(array: &ChunkedArray, mask: FilterMask) -> VortexResult<Vec<ArrayData>> {
    let mut result = Vec::with_capacity(array.nchunks());

    // Pre-materialize the chunk ends for performance.
    // The chunk ends is nchunks+1, which is expected to be in the hundreds or at most thousands.
    let chunk_ends = array.chunk_offsets().into_canonical()?.into_primitive()?;
    let chunk_ends = chunk_ends.maybe_null_slice::<u64>();

    let mut chunk_filters = vec![ChunkFilter::None; array.nchunks()];

    for (slice_start, slice_end) in mask.iter_slices()? {
        let (start_chunk, start_idx) = find_chunk_idx(slice_start, chunk_ends);
        // NOTE: we adjust slice end back by one, in case it ends on a chunk boundary, we do not
        // want to index into the unused chunk.
        let (end_chunk, end_idx) = find_chunk_idx(slice_end - 1, chunk_ends);
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
            let start_chunk_len = chunk_ends[start_chunk + 1] - chunk_ends[start_chunk];
            let start_slice = (start_idx, start_chunk_len as _);
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

    // Now, apply the chunk filter to every slice.
    for (chunk, chunk_filter) in array.chunks().zip(chunk_filters.iter()) {
        match chunk_filter {
            // All => preserve the entire chunk unfiltered.
            ChunkFilter::All => result.push(chunk),
            // None => whole chunk is filtered out, skip
            ChunkFilter::None => {}
            // Slices => turn the slices into a boolean buffer.
            ChunkFilter::Slices(slices) => {
                result.push(filter(&chunk, slices_to_mask(slices, chunk.len()))?);
            }
        }
    }

    Ok(result)
}

/// Filter the chunks using indices.
#[allow(deprecated)]
fn filter_indices(array: &ChunkedArray, mask: FilterMask) -> VortexResult<Vec<ArrayData>> {
    let mut result = Vec::new();
    let mut current_chunk_id = 0;
    let mut chunk_indices = Vec::new();

    // Avoid find_chunk_idx and use our own to avoid the overhead.
    // The array should only be some thousands of values in the general case.
    let chunk_ends = array.chunk_offsets().into_canonical()?.into_primitive()?;
    let chunk_ends = chunk_ends.maybe_null_slice::<u64>();

    for set_index in mask.iter_indices()? {
        let (chunk_id, index) = find_chunk_idx(set_index, chunk_ends);
        if chunk_id != current_chunk_id {
            // Push the chunk we've accumulated.
            if !chunk_indices.is_empty() {
                let chunk = array
                    .chunk(current_chunk_id)
                    .vortex_expect("find_chunk_idx must return valid chunk ID");
                let filtered_chunk = take(
                    chunk,
                    PrimitiveArray::from(chunk_indices.clone()).into_array(),
                    TakeOptions::default(),
                )?;
                result.push(filtered_chunk);
            }

            // Advance the chunk forward, reset the chunk indices buffer.
            current_chunk_id = chunk_id;
            chunk_indices.clear();
        }

        chunk_indices.push(index as u64);
    }

    if !chunk_indices.is_empty() {
        let chunk = array
            .chunk(current_chunk_id)
            .vortex_expect("find_chunk_idx must return valid chunk ID");
        let filtered_chunk = take(
            &chunk,
            PrimitiveArray::from(chunk_indices.clone()).into_array(),
            TakeOptions::default(),
        )?;
        result.push(filtered_chunk);
    }

    Ok(result)
}

// Mirrors the find_chunk_idx method on ChunkedArray, but avoids all of the overhead
// from scalars, dtypes, and metadata cloning.
fn find_chunk_idx(idx: usize, chunk_ends: &[u64]) -> (usize, usize) {
    let chunk_id = chunk_ends
        .search_sorted(&(idx as u64), SearchSortedSide::Right)
        .to_ends_index(chunk_ends.len())
        .saturating_sub(1);
    let chunk_offset = idx - (chunk_ends[chunk_id] as usize);

    (chunk_id, chunk_offset)
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::chunked::compute::filter::slices_to_mask;
    use crate::array::{ChunkedArray, PrimitiveArray};
    use crate::compute::{filter, FilterMask};
    use crate::IntoArrayData;

    #[test]
    fn test_slices_to_predicate() {
        let slices = [(2, 4), (6, 8), (9, 10)];
        let predicate = slices_to_mask(&slices, 11);

        let bools = predicate.to_boolean_buffer().unwrap().iter().collect_vec();

        assert_eq!(
            bools,
            vec![false, false, true, true, false, false, true, true, false, true, false],
        )
    }

    #[test]
    fn filter_chunked_floats() {
        let chunked = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from(vec![f16::from_f32(0.1463623)]).into_array(),
                PrimitiveArray::from(vec![
                    f16::NAN,
                    f16::from_f32(0.24987793),
                    f16::from_f32(0.22497559),
                    f16::from_f32(0.22497559),
                    f16::from_f32(-36160.0),
                ])
                .into_array(),
                PrimitiveArray::from(vec![
                    f16::NAN,
                    f16::NAN,
                    f16::from_f32(0.22497559),
                    f16::from_f32(0.22497559),
                    f16::from_f32(3174.0),
                ])
                .into_array(),
            ],
            DType::Primitive(PType::F16, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();
        let mask = FilterMask::from_iter([
            true, false, false, true, true, true, true, true, true, true, true,
        ]);
        let filtered = filter(&chunked, mask).unwrap();
        assert_eq!(filtered.len(), 9);
    }
}
