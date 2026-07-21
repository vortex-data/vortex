// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

/// Flattens per-chunk take/filter results into a single array.
///
/// Flat dtypes append directly into a canonical builder. Nested dtypes instead collect the chunks
/// and canonicalize them as a chunked array, which reuses the chunks' children zero-copy (e.g.
/// chunked FixedSizeLists canonicalize into one FixedSizeList over the chained elements) and lets
/// the follow-up take push down lazily instead of deep-copying every child.
enum ChunkFlattener {
    Builder(Box<dyn ArrayBuilder>),
    Chunks(Vec<ArrayRef>),
}

impl ChunkFlattener {
    fn new(dtype: &DType, capacity: usize) -> Self {
        if dtype.is_nested() {
            Self::Chunks(Vec::new())
        } else {
            Self::Builder(builder_with_capacity(dtype, capacity))
        }
    }

    fn push(&mut self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        match self {
            Self::Builder(builder) => chunk.append_to_builder(builder.as_mut(), ctx),
            Self::Chunks(chunks) => {
                chunks.push(chunk);
                Ok(())
            }
        }
    }

    fn finish(self, dtype: &DType, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        match self {
            Self::Builder(mut builder) => Ok(builder.finish()),
            // SAFETY: every chunk is a filter or take of a chunk of a chunked array with `dtype`,
            // which leaves the dtype unchanged.
            Self::Chunks(chunks) => unsafe { ChunkedArray::new_unchecked(chunks, dtype.clone()) }
                .into_array()
                .execute::<Canonical>(ctx)
                .map(IntoArray::into_array),
        }
    }
}

fn take_chunked_via_sort(
    array: ArrayView<'_, Chunked>,
    indices: &PrimitiveArray,
    indices_mask: Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices_values = indices.as_slice::<u64>();
    let n = indices_values.len();
    let mut pairs: Vec<(u64, usize)> = indices_values
        .iter()
        .enumerate()
        .filter(|&(position, _)| indices_mask.value(position))
        .map(|(position, &index)| (index, position))
        .collect();
    pairs.sort_unstable();

    if let Some(&(index, _)) = pairs.last() {
        let index = usize::try_from(index)?;
        if index >= array.len() {
            vortex_bail!(OutOfBounds: index, 0, array.len());
        }
    }

    let chunk_offsets = array.chunk_offset_values();
    let nchunks = array.nchunks();
    let mut flattener = ChunkFlattener::new(array.dtype(), pairs.len());
    let mut final_take = BufferMut::<u64>::zeroed(n);
    let mut cursor = 0usize;
    let mut dedup_idx = 0u64;

    for chunk_idx in 0..nchunks {
        let chunk_start = chunk_offsets[chunk_idx];
        let chunk_end = chunk_offsets[chunk_idx + 1];
        let chunk_len = chunk_end - chunk_start;
        let chunk_end_u64 = u64::try_from(chunk_end)?;
        let range_end = cursor + pairs[cursor..].partition_point(|&(v, _)| v < chunk_end_u64);
        let chunk_pairs = &pairs[cursor..range_end];

        if !chunk_pairs.is_empty() {
            let mut local_indices = Vec::new();
            for (i, &(value, original_position)) in chunk_pairs.iter().enumerate() {
                if cursor + i > 0 && value != pairs[cursor + i - 1].0 {
                    dedup_idx += 1;
                }
                let local_index = usize::try_from(value)? - chunk_start;
                if local_indices.last() != Some(&local_index) {
                    local_indices.push(local_index);
                }
                final_take[original_position] = dedup_idx;
            }

            flattener.push(
                array
                    .chunk(chunk_idx)
                    .filter(Mask::from_indices(chunk_len, local_indices))?,
                ctx,
            )?;
        }

        cursor = range_end;
    }

    let flat = flattener.finish(array.dtype(), ctx)?;
    let take_validity = Validity::from_mask(indices_mask, indices.dtype().nullability());
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

fn valid_indices_are_monotonic(indices: &[u64], indices_mask: &Mask) -> bool {
    let mut previous = None;
    for (position, &index) in indices.iter().enumerate() {
        if !indices_mask.value(position) {
            continue;
        }
        if previous.is_some_and(|previous| index < previous) {
            return false;
        }
        previous = Some(index);
    }
    true
}

// TODO(joe): we want to return a chunked array ideally.
fn take_chunked(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices = indices
        .cast(DType::Primitive(PType::U64, indices.dtype().nullability()))?
        .execute::<PrimitiveArray>(ctx)?;

    let indices_mask = indices
        .as_ref()
        .validity()?
        .execute_mask(indices.as_ref().len(), ctx)?;
    let indices_values = indices.as_slice::<u64>();
    let n = indices_values.len();
    let chunk_offsets = array.chunk_offset_values();
    let nchunks = array.nchunks();

    // For a small unsorted fixed-size-list take over a small number of chunks, sorting has lower
    // fixed overhead and lets the nested elements filter in source order. Larger, monotonic, and
    // non-nested takes use bucketing.
    if matches!(array.dtype(), DType::FixedSizeList(..))
        && n <= 64
        && nchunks <= 16
        && !valid_indices_are_monotonic(indices_values, &indices_mask)
    {
        return take_chunked_via_sort(array, &indices, indices_mask, ctx);
    }

    // Route each valid index into its source chunk. Within each bucket, preserve the request order
    // so the taken chunks can be assembled and then restored to the original cross-chunk order.
    let mut buckets = vec![Vec::<(u64, usize)>::new(); nchunks];
    let mut monotonic = true;
    let mut last_index = None;
    let mut sorted_chunk_idx = 0;

    for (original_position, &index) in indices_values.iter().enumerate() {
        if !indices_mask.value(original_position) {
            continue;
        }

        let index = usize::try_from(index)?;
        if index >= array.len() {
            vortex_bail!(OutOfBounds: index, 0, array.len());
        }

        let still_sorted = last_index.is_none_or(|last_index| index >= last_index);
        let chunk_idx = if monotonic && still_sorted {
            while chunk_offsets[sorted_chunk_idx + 1] <= index {
                sorted_chunk_idx += 1;
            }
            sorted_chunk_idx
        } else {
            monotonic = false;
            chunk_offsets.partition_point(|&offset| offset <= index) - 1
        };
        last_index = Some(index);

        let local_index = u64::try_from(index - chunk_offsets[chunk_idx])?;
        buckets[chunk_idx].push((local_index, original_position));
    }

    let mut flattener = ChunkFlattener::new(array.dtype(), indices_mask.true_count());
    let mut final_take =
        (!monotonic || indices.dtype().is_nullable()).then(|| BufferMut::<u64>::zeroed(n));
    let mut grouped_position = 0u64;

    for (chunk_idx, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }

        let mut local_indices = BufferMut::<u64>::with_capacity(bucket.len());
        for (local_index, original_position) in bucket {
            local_indices.push(local_index);
            if let Some(final_take) = &mut final_take {
                final_take[original_position] = grouped_position;
            }
            grouped_position += 1;
        }

        let local_indices =
            PrimitiveArray::new(local_indices.freeze(), Validity::NonNullable).into_array();
        flattener.push(array.chunk(chunk_idx).take(local_indices)?, ctx)?;
    }

    // TODO(joe): can we relax this.
    let flat = flattener.finish(array.dtype(), ctx)?;

    // Non-nullable monotonic indices are already in the same order as the assembled chunks, so no
    // final reorder is needed.
    let Some(final_take) = final_take else {
        return Ok(flat);
    };

    // Restore original order. Carry the original index validity so null indices produce null
    // outputs.
    let take_validity = Validity::from_mask(indices_mask, indices.dtype().nullability());
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

impl TakeExecute for Chunked {
    fn take(
        array: ArrayView<'_, Chunked>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // A single chunk is logically identical to the chunked array, so delegate directly.
        if array.nchunks() == 1 {
            return array.chunk(0).take(indices.clone()).map(Some);
        }

        if let Some(taken) = take_chunked_fsl(array, indices)? {
            return Ok(Some(taken));
        }

        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_piecewise_chunked(array, piecewise_indices, ctx)?
        {
            return Ok(Some(taken));
        }

        take_chunked(array, indices, ctx).map(Some)
    }
}

/// Rewrites take over a chunked array of [`FixedSizeList`] chunks as take over a single
/// [`FixedSizeListArray`] whose elements child chains the chunks' elements.
///
/// Each FSL chunk stores exactly `chunk.len() * list_size` elements starting at its first list,
/// so the chunks' elements children concatenate zero-copy into the elements of the combined
/// array, exactly as chunked canonicalization does. The FSL take implementation then gathers the
/// elements with a single `PiecewiseSequenceArray`, which [`take_piecewise_chunked`] resolves per
/// chunk without expanding one index per element. Chunks that are not FSL-encoded fall back to
/// the generic path since accessing their elements would require canonicalizing whole chunks.
fn take_chunked_fsl(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
) -> VortexResult<Option<ArrayRef>> {
    let DType::FixedSizeList(element_dtype, list_size, _) = array.dtype() else {
        return Ok(None);
    };

    let mut element_chunks = Vec::with_capacity(array.nchunks());
    for chunk in array.iter_chunks() {
        let Some(fsl) = chunk.as_opt::<FixedSizeList>() else {
            return Ok(None);
        };
        element_chunks.push(fsl.elements().clone());
    }

    let validity = array.array().validity()?;
    // SAFETY: every chunk is a FixedSizeList with element dtype `element_dtype`.
    let elements =
        unsafe { ChunkedArray::new_unchecked(element_chunks, element_dtype.as_ref().clone()) }
            .into_array();
    // SAFETY: each FSL chunk holds exactly `chunk.len() * list_size` elements, so the chained
    // elements hold `array.len() * list_size` entries, and the chunked validity covers
    // `array.len()` lists with the array's nullability.
    let fsl = unsafe {
        FixedSizeListArray::new_unchecked(elements, *list_size, validity, array.as_ref().len())
    };
    fsl.into_array().take(indices.clone()).map(Some)
}

/// A per-chunk gather plan: chunk-local sub-piece runs to take from one chunk, in output order.
#[derive(Default)]
struct ChunkGather {
    starts: Vec<u64>,
    lengths: Vec<u64>,
    total: usize,
}

/// Take for [`PiecewiseSequence`] indices with unit multipliers.
///
/// Each piece is a contiguous index run, so instead of expanding one index per element and
/// sorting them (as [`take_chunked`] does), pieces are split at chunk boundaries and gathered
/// from each chunk with a chunk-local `PiecewiseSequenceArray`. When the sub-pieces visit chunks
/// in non-decreasing order the gathered chunks concatenate directly into the result; otherwise
/// the gathered chunks are canonicalized once and a second piecewise take restores output order.
fn take_piecewise_chunked(
    array: ArrayView<'_, Chunked>,
    indices: ArrayView<'_, PiecewiseSequence>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };

    let output_len = indices.as_ref().len();
    let starts: Vec<usize> = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        starts
            .as_slice::<S>()
            .iter()
            .map(|&start| start.as_())
            .collect()
    });
    let lengths: Vec<usize> = match lengths {
        Columnar::Constant(lengths) => vec![constant_unsigned_usize(&lengths); starts.len()],
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                lengths
                    .as_slice::<L>()
                    .iter()
                    .map(|&length| length.as_())
                    .collect()
            })
        }
    };

    let array_len = array.as_ref().len();
    let chunk_offsets = array.chunk_offset_values();
    let nchunks = array.nchunks();

    let mut plans: Vec<ChunkGather> = Vec::new();
    plans.resize_with(nchunks, ChunkGather::default);
    // Sub-pieces in output order: (chunk index, offset within that chunk's gathered output, run
    // length).
    let mut sub_pieces: Vec<(usize, usize, usize)> = Vec::with_capacity(starts.len());
    let mut monotonic = true;
    let mut prev_chunk = 0usize;
    let mut total_len = 0usize;

    for (&start, &length) in starts.iter().zip_eq(&lengths) {
        if length == 0 {
            continue;
        }
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        if end > array_len {
            vortex_bail!(OutOfBounds: end - 1, 0, array_len);
        }
        total_len = total_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;

        // Locate the chunk containing `start`; `<=` skips empty chunks sharing the same offset.
        let mut chunk_idx = chunk_offsets.partition_point(|&offset| offset <= start) - 1;
        let mut cursor = start;
        let mut remaining = length;
        while remaining > 0 {
            while chunk_offsets[chunk_idx + 1] <= cursor {
                chunk_idx += 1;
            }
            let run = remaining.min(chunk_offsets[chunk_idx + 1] - cursor);
            monotonic &= chunk_idx >= prev_chunk;
            prev_chunk = chunk_idx;

            let plan = &mut plans[chunk_idx];
            sub_pieces.push((chunk_idx, plan.total, run));
            plan.starts.push((cursor - chunk_offsets[chunk_idx]) as u64);
            plan.lengths.push(run as u64);
            plan.total += run;

            cursor += run;
            remaining -= run;
        }
    }

    vortex_ensure!(
        total_len == output_len,
        "PiecewiseSequenceArray expanded length {total_len} does not match declared length {output_len}"
    );

    // Gather the sub-pieces of each touched chunk with one chunk-local piecewise take. When the
    // sub-pieces visit chunks in order, the gathered chunks already concatenate into the result.
    if monotonic {
        let mut gathered = Vec::new();
        for (chunk_idx, plan) in plans.into_iter().enumerate() {
            if plan.starts.is_empty() {
                continue;
            }
            let chunk_indices = contiguous_runs(plan.starts, plan.lengths, plan.total);
            gathered.push(array.chunk(chunk_idx).take(chunk_indices)?);
        }
        let result = if gathered.len() == 1 {
            gathered.swap_remove(0)
        } else {
            // SAFETY: every gathered chunk is a take of a chunk with dtype `array.dtype()`, and
            // the non-nullable piecewise indices leave the dtype unchanged.
            unsafe { ChunkedArray::new_unchecked(gathered, array.dtype().clone()) }.into_array()
        };
        return Ok(Some(result));
    }

    // Out-of-order sub-pieces: flatten each chunk's gather once through a [`ChunkFlattener`],
    // then take the sub-piece runs from the flattened result in output order.
    let mut bases = vec![0usize; nchunks];
    let mut running = 0usize;
    let mut flattener = ChunkFlattener::new(array.dtype(), output_len);
    for (chunk_idx, plan) in plans.into_iter().enumerate() {
        if plan.starts.is_empty() {
            continue;
        }
        bases[chunk_idx] = running;
        running += plan.total;
        let chunk_indices = contiguous_runs(plan.starts, plan.lengths, plan.total);
        flattener.push(array.chunk(chunk_idx).take(chunk_indices)?, ctx)?;
    }
    let flat = flattener.finish(array.dtype(), ctx)?;

    let mut reorder_starts = Vec::with_capacity(sub_pieces.len());
    let mut reorder_lengths = Vec::with_capacity(sub_pieces.len());
    for &(chunk_idx, offset, run) in &sub_pieces {
        reorder_starts.push((bases[chunk_idx] + offset) as u64);
        reorder_lengths.push(run as u64);
    }
    flat.take(contiguous_runs(reorder_starts, reorder_lengths, output_len))
        .map(Some)
}

/// Builds a `PiecewiseSequenceArray` of contiguous (unit multiplier) runs whose lengths sum to
/// `total`.
fn contiguous_runs(starts: Vec<u64>, lengths: Vec<u64>, total: usize) -> ArrayRef {
    let count = starts.len();
    debug_assert_eq!(count, lengths.len());
    let starts = PrimitiveArray::new(Buffer::from(starts), Validity::NonNullable).into_array();
    let lengths = match lengths.first() {
        Some(&first) if lengths.iter().all(|&length| length == first) => {
            ConstantArray::new(first, count).into_array()
        }
        _ => PrimitiveArray::new(Buffer::from(lengths), Validity::NonNullable).into_array(),
    };
    let multipliers = ConstantArray::new(1u64, count).into_array();
    // SAFETY: starts, lengths, and multipliers are non-nullable u64 arrays of equal length, and
    // `total` is the sum of the lengths.
    unsafe { PiecewiseSequenceArray::new_unchecked(starts, lengths, multipliers, total) }
        .into_array()
}

#[cfg(test)]
mod test {
    use vortex_buffer::Buffer;
    use vortex_buffer::bitbuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PiecewiseSequenceArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    fn chunked_i32() -> VortexResult<ChunkedArray> {
        ChunkedArray::try_new(
            vec![
                buffer![0i32, 1, 2, 3, 4].into_array(),
                buffer![5i32, 6, 7, 8, 9].into_array(),
                buffer![10i32, 11, 12, 13, 14].into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
    }

    fn contiguous_pieces(starts: &[u64], lengths: &[u64]) -> VortexResult<ArrayRef> {
        let len = usize::try_from(lengths.iter().sum::<u64>())?;
        Ok(PiecewiseSequenceArray::try_new(
            starts.iter().copied().collect::<Buffer<u64>>().into_array(),
            lengths
                .iter()
                .copied()
                .collect::<Buffer<u64>>()
                .into_array(),
            ConstantArray::new(1u64, starts.len()).into_array(),
            len,
        )?
        .into_array())
    }

    #[test]
    fn test_take_piecewise_monotonic_spanning_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = chunked_i32()?;

        // The second piece crosses the first chunk boundary, the third spans the last two chunks.
        let indices = contiguous_pieces(&[1, 4, 9], &[3, 4, 6])?;
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_piecewise_interleaved() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = chunked_i32()?;

        // Pieces visit chunks out of order, forcing the reorder take.
        let indices = contiguous_pieces(&[12, 2, 7, 0], &[3, 2, 3, 1])?;
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([12i32, 13, 14, 2, 3, 7, 8, 9, 0]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_piecewise_whole_array() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = chunked_i32()?;

        let indices = contiguous_pieces(&[0], &[15])?;
        let result = arr.take(indices)?;

        assert_arrays_eq!(result, PrimitiveArray::from_iter(0i32..15), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_piecewise_across_empty_chunk() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = ChunkedArray::try_new(
            vec![
                buffer![0i32, 1, 2, 3, 4].into_array(),
                PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
                buffer![5i32, 6, 7, 8, 9].into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;

        let indices = contiguous_pieces(&[3], &[4])?;
        let result = arr.take(indices)?;

        assert_arrays_eq!(result, PrimitiveArray::from_iter([3i32, 4, 5, 6]), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_piecewise_zero_length_pieces() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = chunked_i32()?;

        let indices = contiguous_pieces(&[5, 0, 2], &[0, 4, 0])?;
        let result = arr.take(indices)?;

        assert_arrays_eq!(result, PrimitiveArray::from_iter([0i32, 1, 2, 3]), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_piecewise_out_of_bounds() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = chunked_i32()?;

        let indices = contiguous_pieces(&[12], &[5])?;
        let result = arr
            .take(indices)
            .and_then(|taken| taken.execute::<Canonical>(&mut ctx));

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_take_piecewise_non_unit_multiplier() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = chunked_i32()?;

        // Multiplier 2 falls back to the generic path but must stay correct.
        let indices = PiecewiseSequenceArray::try_new(
            buffer![0u64, 1].into_array(),
            buffer![5u64, 3].into_array(),
            buffer![2u64, 4].into_array(),
            8,
        )?
        .into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([0i32, 2, 4, 6, 8, 1, 5, 9]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_piecewise_nullable_values() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([Some(0i32), None, Some(2)]).into_array(),
                PrimitiveArray::from_option_iter([None, Some(4i32), Some(5)]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::Nullable),
        )?;

        let indices = contiguous_pieces(&[4, 1], &[2, 3])?;
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(4i32), Some(5), None, Some(2), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_fsl_over_chunked_elements() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Chunk boundaries at 8 and 13 do not line up with the list size of 3.
        let elements = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter(0i32..8).into_array(),
                PrimitiveArray::from_iter(8i32..13).into_array(),
                PrimitiveArray::from_iter(13i32..18).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();
        let fsl = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 6)?;

        let indices = buffer![4u64, 0, 2, 5, 1, 4].into_array();
        let result = fsl.take(indices.clone())?;
        let expected = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..18).into_array(),
            3,
            Validity::NonNullable,
            6,
        )?
        .take(indices)?;

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_fsl_over_chunked_elements_conformance() -> VortexResult<()> {
        let elements = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter(0i32..8).into_array(),
                PrimitiveArray::from_iter(8i32..13).into_array(),
                PrimitiveArray::from_iter(13i32..18).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();
        let fsl = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 6)?;
        test_take_conformance(
            &fsl.into_array(),
            &mut array_session().create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn test_take_chunked_fsl() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..6).into_array(),
            2,
            Validity::NonNullable,
            3,
        )?;
        let c1 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(6i32..10).into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;
        let dtype = c0.dtype().clone();
        let arr = ChunkedArray::try_new(vec![c0.into_array(), c1.into_array()], dtype)?;

        let indices = buffer![4u64, 0, 3, 3, 1].into_array();
        let result = arr.take(indices.clone())?;
        let expected = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..10).into_array(),
            2,
            Validity::NonNullable,
            5,
        )?
        .take(indices)?;

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_chunked_fsl_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..6).into_array(),
            2,
            Validity::Array(bitbuffer![1 0 1].into_array()),
            3,
        )?;
        let c1 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(6i32..10).into_array(),
            2,
            Validity::AllValid,
            2,
        )?;
        let dtype = c0.dtype().clone();
        let arr = ChunkedArray::try_new(vec![c0.into_array(), c1.into_array()], dtype)?;

        let indices =
            PrimitiveArray::from_option_iter([Some(4u64), None, Some(0), Some(1)]).into_array();
        let result = arr.take(indices.clone())?;
        let expected = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..10).into_array(),
            2,
            Validity::Array(bitbuffer![1 0 1 1 1].into_array()),
            5,
        )?
        .take(indices)?;

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_chunked_fsl_non_fsl_chunk_falls_back() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..6).into_array(),
            2,
            Validity::NonNullable,
            3,
        )?;
        let c1 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(6i32..10).into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;
        let dtype = c0.dtype().clone();
        // Wrap one chunk in a nested chunked array so it is not FSL-encoded.
        let nested = ChunkedArray::try_new(vec![c1.into_array()], dtype.clone())?;
        let arr = ChunkedArray::try_new(vec![c0.into_array(), nested.into_array()], dtype)?;

        let indices = buffer![4u64, 0, 3, 1].into_array();
        let result = arr.take(indices.clone())?;
        let expected = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..10).into_array(),
            2,
            Validity::NonNullable,
            5,
        )?
        .take(indices)?;

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_chunked_struct_nested_flatten() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let s0 = StructArray::try_new(
            ["a"].into(),
            vec![buffer![0i32, 1, 2].into_array()],
            3,
            Validity::NonNullable,
        )?;
        let s1 = StructArray::try_new(
            ["a"].into(),
            vec![buffer![3i32, 4].into_array()],
            2,
            Validity::NonNullable,
        )?;
        let dtype = s0.dtype().clone();
        let arr = ChunkedArray::try_new(vec![s0.into_array(), s1.into_array()], dtype)?;

        let result = arr.take(buffer![4u64, 0, 2, 4].into_array())?;
        let expected = StructArray::try_new(
            ["a"].into(),
            vec![buffer![4i32, 0, 2, 4].into_array()],
            4,
            Validity::NonNullable,
        )?;

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_chunked_fsl_conformance() -> VortexResult<()> {
        let c0 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(0i32..6).into_array(),
            2,
            Validity::NonNullable,
            3,
        )?;
        let c1 = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(6i32..10).into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;
        let dtype = c0.dtype().clone();
        let arr = ChunkedArray::try_new(vec![c0.into_array(), c1.into_array()], dtype)?;
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn test_take() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = buffer![0u64, 0, 6, 4].into_array();

        let result = arr.take(indices).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 1, 1, 2]), &mut ctx);
    }

    #[test]
    fn test_take_nullable_values() {
        let mut ctx = array_session().create_execution_ctx();
        let a = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = PrimitiveArray::new(buffer![0u64, 0, 6, 4], Validity::NonNullable);

        let result = arr.take(indices.into_array()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([1i32, 1, 1, 2].map(Some)),
            &mut ctx
        );
    }

    #[test]
    fn test_take_nullable_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = PrimitiveArray::new(
            buffer![0u64, 0, 6, 4],
            Validity::Array(bitbuffer![1 0 0 1].into_array()),
        );

        let result = arr.take(indices.into_array()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(1i32), None, None, Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_take_nullable_struct() {
        let mut ctx = array_session().create_execution_ctx();
        let struct_array =
            StructArray::try_new(FieldNames::default(), vec![], 100, Validity::NonNullable)
                .unwrap();

        let arr = ChunkedArray::from_iter(vec![
            struct_array.clone().into_array(),
            struct_array.into_array(),
        ]);

        let result = arr
            .take(PrimitiveArray::from_option_iter(vec![Some(0), None, Some(101)]).into_array())
            .unwrap();

        let expect = StructArray::try_new(
            FieldNames::default(),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array()),
        )
        .unwrap();
        assert_arrays_eq!(result, expect, &mut ctx);
    }

    #[test]
    fn test_empty_take() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);

        let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
        let result = arr.take(indices.into_array()).unwrap();

        assert!(result.is_empty());
        assert_eq!(result.dtype(), arr.dtype());
        assert_arrays_eq!(
            result,
            PrimitiveArray::empty::<i32>(Nullability::NonNullable),
            &mut ctx
        );
    }

    #[test]
    fn test_take_shuffled_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![0i32, 1, 2].into_array();
        let c1 = buffer![3i32, 4, 5].into_array();
        let c2 = buffer![6i32, 7, 8].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1, c2],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Fully shuffled indices that cross every chunk boundary.
        let indices = buffer![8u64, 0, 5, 3, 2, 7, 1, 6, 4].into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([8i32, 0, 5, 3, 2, 7, 1, 6, 4]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_shuffled_duplicates_with_empty_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let empty = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
        let arr = ChunkedArray::try_new(
            vec![
                empty.clone(),
                buffer![0i32, 1].into_array(),
                empty.clone(),
                buffer![2i32, 3].into_array(),
                empty,
            ],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        let result = arr.take(buffer![3u64, 0, 2, 0, 3, 1, 2].into_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([3i32, 0, 2, 0, 3, 1, 2]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_small_shuffled_fixed_size_lists() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let first = FixedSizeListArray::new(
            buffer![0i32, 1, 2, 3].into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let second = FixedSizeListArray::new(
            buffer![4i32, 5, 6, 7].into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let dtype = first.dtype().clone();
        let array = ChunkedArray::try_new(vec![first, second], dtype)?;

        let result = array.take(buffer![3u64, 0, 2, 1, 3].into_array())?;
        let expected = FixedSizeListArray::new(
            buffer![6i32, 7, 0, 1, 4, 5, 2, 3, 6, 7].into_array(),
            2,
            Validity::NonNullable,
            5,
        );

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_shuffled_large() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let nchunks: i32 = 100;
        let chunk_len: i32 = 1_000;
        let total = nchunks * chunk_len;

        let chunks: Vec<_> = (0..nchunks)
            .map(|c| {
                let start = c * chunk_len;
                PrimitiveArray::from_iter(start..start + chunk_len).into_array()
            })
            .collect();
        let dtype = chunks[0].dtype().clone();
        let arr = ChunkedArray::try_new(chunks, dtype)?;

        // Fisher-Yates shuffle with a fixed seed for determinism.
        let mut indices: Vec<u64> = (0..u64::try_from(total)?).collect();
        let mut seed: u64 = 0xdeadbeef;
        for i in (1..indices.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let indices_arr = PrimitiveArray::new(Buffer::from(indices.clone()), Validity::NonNullable);
        let result = arr.take(indices_arr.into_array())?;

        // Verify every element.
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;
        let result_vals = result.as_slice::<i32>();
        for (pos, &idx) in indices.iter().enumerate() {
            assert_eq!(
                result_vals[pos],
                i32::try_from(idx)?,
                "mismatch at position {pos}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_take_null_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![10i32, 20, 30].into_array();
        let c1 = buffer![40i32, 50, 60].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Indices with nulls scattered across chunk boundaries.
        let indices =
            PrimitiveArray::from_option_iter([Some(5u64), None, Some(0), Some(3), None, Some(2)]);
        let result = arr.take(indices.into_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([
                Some(60i32),
                None,
                Some(10),
                Some(40),
                None,
                Some(30)
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_chunked_conformance() {
        let a = buffer![1i32, 2, 3].into_array();
        let b = buffer![4i32, 5].into_array();
        let arr = ChunkedArray::try_new(
            vec![a, b],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )
        .unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );

        // Test with nullable chunked array
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let b = PrimitiveArray::from_option_iter([Some(4i32), Some(5)]);
        let dtype = a.dtype().clone();
        let arr = ChunkedArray::try_new(vec![a.into_array(), b.into_array()], dtype).unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );

        // Test with multiple identical chunks
        let chunk = buffer![10i32, 20, 30, 40, 50].into_array();
        let arr = ChunkedArray::try_new(
            vec![chunk.clone(), chunk.clone(), chunk.clone()],
            chunk.dtype().clone(),
        )
        .unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
