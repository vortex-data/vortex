// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::Slice;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::contiguous_runs;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::struct_::StructArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::executor::Executable;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::optimizer::ArrayOptimizer;
use crate::validity::Validity;

/// Flattens per-chunk take/filter results into a single array. Flat dtypes append directly into
/// a canonical builder; nested dtypes collect the chunks and canonicalize them as a chunked
/// array, which reuses the chunks' children zero-copy.
enum ChunkFlattener {
    Builder(Box<dyn ArrayBuilder>),
    Chunks(Vec<ArrayRef>),
}

impl ChunkFlattener {
    fn new(dtype: &DType, capacity: usize, allocator: &BufferAllocatorRef) -> Self {
        if dtype.is_nested() {
            Self::Chunks(Vec::new())
        } else {
            Self::Builder(builder_with_capacity_in(dtype, capacity, allocator))
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
            vortex_bail!(
                "index {} out of bounds from {} to {}",
                index,
                0,
                array.len()
            );
        }
    }

    let chunk_offsets = array.chunk_offset_values();
    let nchunks = array.nchunks();
    let mut flattener = ChunkFlattener::new(array.dtype(), pairs.len(), ctx.allocator());
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

    // Strictly increasing non-nullable indices select every row at most once and in order, so the
    // per-chunk gathers concatenate straight into the result with no dedup or reorder take.
    if !indices.as_ref().dtype().is_nullable() && indices_values.is_sorted_by(|a, b| a < b) {
        return take_chunked_sorted(array, indices_values);
    }

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
            vortex_bail!(
                "index {} out of bounds from {} to {}",
                index,
                0,
                array.len()
            );
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

    let mut flattener =
        ChunkFlattener::new(array.dtype(), indices_mask.true_count(), ctx.allocator());
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

    // A single take restores original order and expands duplicates. Carrying the original index
    // validity makes null indices produce null outputs.
    let take_validity = Validity::from_mask(indices_mask, indices.dtype().nullability());
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

/// Take for strictly increasing, non-nullable indices: every row is selected at most once and in
/// order, so the per-chunk gathers concatenate directly into the result with no bucketing, dedup,
/// or reorder take.
fn take_chunked_sorted(
    array: ArrayView<'_, Chunked>,
    indices_values: &[u64],
) -> VortexResult<ArrayRef> {
    let chunk_offsets = array.chunk_offset_values();
    let mut chunks = Vec::new();
    let mut cursor = 0usize;

    // Skip straight to the chunk holding the first index instead of walking every leading chunk;
    // `<=` steps past empty chunks sharing the same offset.
    let first_chunk = match indices_values.first() {
        Some(&first) => {
            let first = usize::try_from(first)?;
            chunk_offsets
                .partition_point(|&offset| offset <= first)
                .saturating_sub(1)
        }
        None => 0,
    };

    for chunk_idx in first_chunk..array.nchunks() {
        if cursor == indices_values.len() {
            break;
        }
        let chunk_start = chunk_offsets[chunk_idx];
        let chunk_end = chunk_offsets[chunk_idx + 1];
        let chunk_end_u64 = u64::try_from(chunk_end)?;

        let range_end = cursor + indices_values[cursor..].partition_point(|&v| v < chunk_end_u64);
        if range_end > cursor {
            let chunk_start_u64 = u64::try_from(chunk_start)?;
            let mut local_indices = BufferMut::<u64>::with_capacity(range_end - cursor);
            for &val in &indices_values[cursor..range_end] {
                local_indices.push(val - chunk_start_u64);
            }
            let local_indices =
                PrimitiveArray::new(local_indices.freeze(), Validity::NonNullable).into_array();
            chunks.push(array.chunk(chunk_idx).take(local_indices)?);
        }
        cursor = range_end;
    }
    if cursor != indices_values.len() {
        vortex_bail!(
            "index {} out of bounds from {} to {}",
            usize::try_from(indices_values[cursor])?,
            0,
            array.as_ref().len()
        );
    }

    if chunks.len() == 1 {
        return Ok(chunks.swap_remove(0));
    }
    // SAFETY: every chunk is a take of a chunk of `array` with non-nullable indices, which
    // leaves the dtype unchanged.
    Ok(unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()) }.into_array())
}

impl TakeExecute for Chunked {
    fn take(
        array: ArrayView<'_, Chunked>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.nchunks() == 1 {
            return array.chunk(0).take(indices.clone()).map(Some);
        }

        if let Some(taken) = take_chunked_fsl(array, indices, ctx)? {
            return Ok(Some(taken));
        }

        if let Some(taken) = take_chunked_struct(array, indices, ctx)? {
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

/// Rewrites take over a chunked fixed-size list array as take over a single
/// [`FixedSizeListArray`]. The swizzle is structural: FSL-encoded chunks execute to themselves,
/// so their elements chain zero-copy, and the FSL take then gathers per-chunk element runs
/// instead of taking each chunk.
fn take_chunked_fsl(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !matches!(array.dtype(), DType::FixedSizeList(..)) {
        return Ok(None);
    }

    let Some((array, indices)) = touched_chunks(array, indices, ctx)? else {
        let fsl = array.as_ref().clone().execute::<FixedSizeListArray>(ctx)?;
        return fsl.into_array().take(indices.clone()).map(Some);
    };

    take_swizzled::<FixedSizeListArray>(array, indices, ctx).map(Some)
}

/// Rewrites take over a chunked struct array with a fixed-size-list field as take over a single
/// [`StructArray`] whose fields are chunked. The swizzle is structural: it needs no decode when
/// every chunk is already a struct, which is the case for the chunks of a file scan. Each
/// fixed-size-list field then takes once with the element-run gather of [`take_chunked_fsl`],
/// instead of a struct take and a list take for each chunk that the indices touch.
///
/// Structs without a fixed-size-list field keep the per-chunk take: there, each field would route
/// the indices to the chunks again, which costs more than the struct take of each chunk.
fn take_chunked_struct(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let DType::Struct(fields, _) = array.dtype() else {
        return Ok(None);
    };

    if !fields
        .fields()
        .any(|field| matches!(field, DType::FixedSizeList(..)))
    {
        return Ok(None);
    }

    // A chunk with another encoding would have to execute to a struct, which can decode the whole
    // chunk to take a few rows of it. Only the chunks that the swizzle visits must be structs.
    match touched_chunks(array, indices, ctx)? {
        Some((chunks, indices)) => {
            if !chunks.iter().all(is_swizzle_ready_struct) {
                return Ok(None);
            }
            take_swizzled::<StructArray>(chunks, indices, ctx).map(Some)
        }
        None => {
            if !array.iter_chunks().all(is_swizzle_ready_struct) {
                return Ok(None);
            }
            let structs = array.as_ref().clone().execute::<StructArray>(ctx)?;
            structs.into_array().take(indices.clone()).map(Some)
        }
    }
}

/// Returns true if `chunk` is a struct whose fixed-size-list fields are all FSL-encoded.
///
/// The list take of the swizzled struct executes each chunk of a fixed-size-list field to a
/// [`FixedSizeListArray`] before it selects rows. For another encoding, for example a dictionary,
/// this decodes all rows of the chunk. The per-chunk take selects the rows first, for example only
/// the dictionary codes, so these chunks use it.
fn is_swizzle_ready_struct(chunk: &ArrayRef) -> bool {
    let Some(struct_chunk) = chunk.as_opt::<Struct>() else {
        return false;
    };

    struct_chunk
        .struct_fields()
        .fields()
        .enumerate()
        .filter(|(_, dtype)| matches!(dtype, DType::FixedSizeList(..)))
        .all(|(idx, _)| struct_chunk.unmasked_field(idx).is::<FixedSizeList>())
}

/// Takes `indices` from `chunks` through one swizzled array of type `A`, or directly from the
/// chunk when there is only one.
fn take_swizzled<A>(
    chunks: Vec<ArrayRef>,
    indices: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    A: Executable + IntoArray,
{
    if let [chunk] = chunks.as_slice() {
        return chunk.take(indices);
    }

    let dtype = chunks[0].dtype().clone();
    // SAFETY: the chunks are chunks of one chunked array, so they all have its dtype.
    let swizzled = unsafe { ChunkedArray::new_unchecked(chunks, dtype) }
        .into_array()
        .execute::<A>(ctx)?;
    swizzled.into_array().take(indices)
}

/// The chunks that the non-nullable `indices` touch, in chunk order, and the indices into them.
///
/// A swizzle of a chunked array visits every chunk, so a small take over many chunks first keeps
/// only the chunks that it reads. Returns `None` for nullable indices and for indices that touch
/// every chunk.
fn touched_chunks(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(Vec<ArrayRef>, ArrayRef)>> {
    if indices.dtype().is_nullable() || indices.is_empty() {
        return Ok(None);
    }

    if let Some(piecewise) = indices.as_opt::<PiecewiseSequence>()
        && let Some((starts, lengths)) = maybe_contiguous_slices(piecewise, ctx)?
    {
        return touched_chunks_piecewise(array, piecewise, &starts, lengths);
    }

    let indices = indices
        .cast(DType::Primitive(PType::U64, indices.dtype().nullability()))?
        .execute::<PrimitiveArray>(ctx)?;
    let indices = indices.as_slice::<u64>();
    let chunk_offsets = array.chunk_offset_values();
    let array_len = array.as_ref().len();

    // `<=` steps past empty chunks that share the offset of the next chunk.
    let index_chunks = indices
        .iter()
        .map(|&index| -> VortexResult<usize> {
            let index = usize::try_from(index)?;
            if index >= array_len {
                vortex_bail!("index {} out of bounds from {} to {}", index, 0, array_len);
            }
            Ok(chunk_offsets.partition_point(|&offset| offset <= index) - 1)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let mut touched = index_chunks.clone();
    touched.sort_unstable();
    touched.dedup();

    if touched.len() == array.nchunks() {
        return Ok(None);
    }

    let touched_offsets = touched_chunk_offsets(chunk_offsets, &touched);
    let remapped = indices
        .iter()
        .zip_eq(index_chunks)
        .map(|(&index, chunk_idx)| -> VortexResult<u64> {
            let position = touched
                .binary_search(&chunk_idx)
                .map_err(|_| vortex_err!("chunk {chunk_idx} is not a touched chunk"))?;
            let local = usize::try_from(index)? - chunk_offsets[chunk_idx];
            Ok(u64::try_from(touched_offsets[position] + local)?)
        })
        .collect::<VortexResult<Buffer<u64>>>()?;

    let chunks = touched
        .iter()
        .map(|&chunk_idx| array.chunk(chunk_idx).clone())
        .collect();
    Ok(Some((
        chunks,
        PrimitiveArray::new(remapped, Validity::NonNullable).into_array(),
    )))
}

/// Computes touched chunks and remapped [`PiecewiseSequence`] indices directly from contiguous
/// runs without expanding them.
fn touched_chunks_piecewise(
    array: ArrayView<'_, Chunked>,
    indices: ArrayView<'_, PiecewiseSequence>,
    starts: &PrimitiveArray,
    lengths: Columnar,
) -> VortexResult<Option<(Vec<ArrayRef>, ArrayRef)>> {
    let output_len = indices.as_ref().len();
    let sub_pieces = split_piecewise_runs(array, starts, lengths, output_len)?;

    let mut touched: Vec<usize> = sub_pieces
        .iter()
        .map(|&(chunk_idx, ..)| chunk_idx)
        .collect();
    touched.sort_unstable();
    touched.dedup();
    if touched.len() == array.nchunks() {
        return Ok(None);
    }

    let touched_offsets = touched_chunk_offsets(array.chunk_offset_values(), &touched);
    let mut remapped_starts = Vec::with_capacity(sub_pieces.len());
    let mut remapped_lengths = Vec::with_capacity(sub_pieces.len());
    for (chunk_idx, local_start, run) in sub_pieces {
        let position = touched
            .binary_search(&chunk_idx)
            .map_err(|_| vortex_err!("chunk {chunk_idx} is not a touched chunk"))?;
        let remapped_start = u64::try_from(touched_offsets[position] + local_start)?;
        let remapped_len = u64::try_from(run)?;
        match (remapped_starts.last(), remapped_lengths.last_mut()) {
            (Some(&prev_start), Some(prev_len)) if prev_start + *prev_len == remapped_start => {
                *prev_len += remapped_len;
            }
            _ => {
                remapped_starts.push(remapped_start);
                remapped_lengths.push(remapped_len);
            }
        }
    }

    let chunks = touched
        .iter()
        .map(|&chunk_idx| array.chunk(chunk_idx).clone())
        .collect();
    Ok(Some((
        chunks,
        contiguous_runs(remapped_starts, remapped_lengths, output_len),
    )))
}

/// The offset of each touched chunk in the compacted touched-chunks array.
fn touched_chunk_offsets(chunk_offsets: &[usize], touched: &[usize]) -> Vec<usize> {
    let mut touched_offsets = Vec::with_capacity(touched.len());
    let mut offset = 0usize;
    for &chunk_idx in touched {
        touched_offsets.push(offset);
        offset += chunk_offsets[chunk_idx + 1] - chunk_offsets[chunk_idx];
    }
    touched_offsets
}

fn unsigned_slice_as_usize<T: UnsignedPType>(array: &PrimitiveArray) -> Vec<usize> {
    array
        .as_slice::<T>()
        .iter()
        .map(|&value| value.as_())
        .collect()
}

fn contiguous_slice_ranges(starts: &PrimitiveArray, lengths: Columnar) -> (Vec<usize>, Vec<usize>) {
    let starts = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        unsigned_slice_as_usize::<S>(starts)
    });
    let lengths = match lengths {
        Columnar::Constant(lengths) => vec![constant_unsigned_usize(&lengths); starts.len()],
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                unsigned_slice_as_usize::<L>(&lengths)
            })
        }
    };
    (starts, lengths)
}

/// Splits contiguous [`PiecewiseSequence`] runs at chunk boundaries into
/// `(chunk_idx, local_start, run)` sub-pieces in output order, validating bounds and total length.
fn split_piecewise_runs(
    array: ArrayView<'_, Chunked>,
    starts: &PrimitiveArray,
    lengths: Columnar,
    output_len: usize,
) -> VortexResult<Vec<(usize, usize, usize)>> {
    let (starts, lengths) = contiguous_slice_ranges(starts, lengths);
    let chunk_offsets = array.chunk_offset_values();
    let array_len = array.as_ref().len();

    let mut pieces = Vec::with_capacity(starts.len());
    let mut total_len = 0usize;

    for (&start, &length) in starts.iter().zip_eq(&lengths) {
        if length == 0 {
            continue;
        }
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        if end > array_len {
            vortex_bail!(
                "index {} out of bounds from {} to {}",
                end - 1,
                0,
                array_len
            );
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
            pieces.push((chunk_idx, cursor - chunk_offsets[chunk_idx], run));
            cursor += run;
            remaining -= run;
        }
    }

    vortex_ensure_eq!(
        total_len,
        output_len,
        "PiecewiseSequenceArray expanded length does not match declared length",
    );

    Ok(pieces)
}

/// Out-of-order runs at least this long on average append to the result directly, in output
/// order.
///
/// Each run is a slice of a chunk that appends into one builder, so the rows are decoded once.
/// Each slice has a fixed cost, so for very short runs a second copy of the gathered rows costs
/// about the same. In the `chunked_fsl_take` bench, the two paths are equal for runs of 16 rows,
/// and the direct path is faster from 64 rows.
const MIN_AVERAGE_DIRECT_RUN: usize = 32;

/// Runs of a slice that does its work at execution append directly only if they are at least
/// this long on average.
const MIN_AVERAGE_EXECUTED_RUN: usize = 256;

/// Runs of a slice that does its work at execution append directly only if they read at most
/// `1 / SPARSE_CHUNK_RATIO` of the rows of each chunk.
const SPARSE_CHUNK_RATIO: usize = 8;

/// A per-chunk gather plan: chunk-local sub-piece runs to take from one chunk, in output order.
#[derive(Default)]
struct ChunkGather {
    starts: Vec<u64>,
    lengths: Vec<u64>,
    total: usize,
}

/// Take for [`PiecewiseSequence`] indices with unit multipliers.
///
/// Each piece is a contiguous index run, so instead of expanding one index per element, pieces
/// are split at chunk boundaries and gathered from each chunk with a chunk-local
/// `PiecewiseSequenceArray`. Sub-pieces that visit chunks in non-decreasing order concatenate
/// directly into the result; otherwise the gathered chunks are flattened once and a second
/// piecewise take restores output order.
fn take_piecewise_chunked(
    array: ArrayView<'_, Chunked>,
    indices: ArrayView<'_, PiecewiseSequence>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };

    let output_len = indices.as_ref().len();
    let pieces = split_piecewise_runs(array, &starts, lengths, output_len)?;

    // Plans exist only for the chunks that the pieces touch, in the order of the first visit, so
    // the cost scales with the number of pieces and not with the number of chunks.
    let mut plans: Vec<(usize, ChunkGather)> = Vec::new();
    let mut plan_of_chunk: HashMap<usize, usize> = HashMap::default();
    // Sub-pieces in output order: (plan index, offset within that chunk's gathered output, run
    // length).
    let mut sub_pieces: Vec<(usize, usize, usize)> = Vec::with_capacity(pieces.len());
    // Sub-pieces in output order as (chunk index, start in the chunk, run length). A run that
    // continues the previous run in the same chunk extends it.
    let mut runs: Vec<(usize, usize, usize)> = Vec::with_capacity(pieces.len());
    let mut monotonic = true;
    let mut prev_chunk = 0usize;

    for (chunk_idx, local_start, run) in pieces {
        monotonic &= chunk_idx >= prev_chunk;
        prev_chunk = chunk_idx;

        // Consecutive sub-pieces usually stay in one chunk, so check the last plan first.
        let plan_idx = match plans.last() {
            Some((last_chunk, _)) if *last_chunk == chunk_idx => plans.len() - 1,
            _ => *plan_of_chunk.entry(chunk_idx).or_insert_with(|| {
                plans.push((chunk_idx, ChunkGather::default()));
                plans.len() - 1
            }),
        };
        let plan = &mut plans[plan_idx].1;

        // Consecutive sub-pieces in the same chunk stay adjacent in its gathered output, so
        // they merge into one reorder run.
        match sub_pieces.last_mut() {
            Some((last_plan, _, last_run)) if *last_plan == plan_idx => *last_run += run,
            _ => sub_pieces.push((plan_idx, plan.total, run)),
        }
        // A run that continues exactly where the chunk's previous run ended extends it, so
        // contiguous spans gather as one run.
        match runs.last_mut() {
            Some((last_chunk, last_start, last_run))
                if *last_chunk == chunk_idx && *last_start + *last_run == local_start =>
            {
                *last_run += run;
            }
            _ => runs.push((chunk_idx, local_start, run)),
        }

        let local_start = local_start as u64;
        match (plan.starts.last(), plan.lengths.last_mut()) {
            (Some(&prev_start), Some(prev_len)) if prev_start + *prev_len == local_start => {
                *prev_len += run as u64;
            }
            _ => {
                plan.starts.push(local_start);
                plan.lengths.push(run as u64);
            }
        }
        plan.total += run;
    }

    // Chunks visited in order: the plans are in chunk order, and the per-chunk gathers already
    // concatenate into the result.
    if monotonic {
        let mut gathered = Vec::with_capacity(plans.len());
        for (chunk_idx, plan) in plans {
            gathered.push(gather_chunk(array.chunk(chunk_idx), plan)?);
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

    // Long out-of-order runs: append a slice of each run to the result in output order, so the
    // rows are decoded once and copied once.
    if !array.dtype().is_nested()
        && output_len >= runs.len() * MIN_AVERAGE_DIRECT_RUN
        && (sparse_long_runs(array, &plans, runs.len(), output_len)
            || slices_reduce(array, &runs, ctx)?)
    {
        let mut flattener = ChunkFlattener::new(array.dtype(), output_len, ctx.allocator());

        for &(chunk_idx, local_start, run) in &runs {
            flattener.push(
                array
                    .chunk(chunk_idx)
                    .slice(local_start..local_start + run)?,
                ctx,
            )?;
        }

        return flattener.finish(array.dtype(), ctx).map(Some);
    }

    // Out-of-order sub-pieces: flatten the per-chunk gathers, then take the sub-piece runs from
    // the flattened result in output order.
    let mut bases = Vec::with_capacity(plans.len());
    let mut running = 0usize;
    let mut flattener = ChunkFlattener::new(array.dtype(), output_len, ctx.allocator());
    for (chunk_idx, plan) in plans {
        bases.push(running);
        running += plan.total;
        flattener.push(gather_chunk(array.chunk(chunk_idx), plan)?, ctx)?;
    }
    let flat = flattener.finish(array.dtype(), ctx)?;

    let mut reorder_starts = Vec::with_capacity(sub_pieces.len());
    let mut reorder_lengths = Vec::with_capacity(sub_pieces.len());
    for &(plan_idx, offset, run) in &sub_pieces {
        reorder_starts.push((bases[plan_idx] + offset) as u64);
        reorder_lengths.push(run as u64);
    }
    flat.take(contiguous_runs(reorder_starts, reorder_lengths, output_len))
        .map(Some)
}

/// Tells if the runs are long and read a small part of each chunk.
///
/// Then a slice of each run reads less than the take of the runs of each chunk, also when the
/// slice does its work at execution. In the `alp_fsl_take` bench, which has patches, a take of
/// rows of 768 elements is up to 7 times faster this way, and a take of rows of 64 elements is
/// 20% slower.
fn sparse_long_runs(
    array: ArrayView<'_, Chunked>,
    plans: &[(usize, ChunkGather)],
    num_runs: usize,
    output_len: usize,
) -> bool {
    output_len >= num_runs * MIN_AVERAGE_EXECUTED_RUN
        && plans.iter().all(|(chunk_idx, plan)| {
            plan.total.saturating_mul(SPARSE_CHUNK_RATIO) <= array.chunk(*chunk_idx).len()
        })
}

/// Tells if a slice of each chunk that `runs` read becomes an encoded array with no execution.
///
/// Such a slice decodes only its rows when it appends to a builder. A slice that does not
/// reduce, such as a slice of an array with patches, does its work for each run, and a take of
/// the runs of each chunk does it once for the chunk. The first run of each chunk is the test.
fn slices_reduce(
    array: ArrayView<'_, Chunked>,
    runs: &[(usize, usize, usize)],
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mut tested = vec![false; array.nchunks()];

    for &(chunk_idx, start, run) in runs {
        if std::mem::replace(&mut tested[chunk_idx], true) {
            continue;
        }

        let slice = array
            .chunk(chunk_idx)
            .slice(start..start + run)?
            .optimize_ctx(ctx)?;

        if slice.is::<Slice>() {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Gathers a chunk's planned runs: a single run is a zero-copy slice, multiple runs become a
/// chunk-local piecewise take.
fn gather_chunk(chunk: &ArrayRef, plan: ChunkGather) -> VortexResult<ArrayRef> {
    if let [start] = plan.starts.as_slice() {
        let start = usize::try_from(*start)?;
        return chunk.slice(start..start + plan.total);
    }
    chunk.take(contiguous_runs(plan.starts, plan.lengths, plan.total))
}

#[cfg(test)]
mod tests;
