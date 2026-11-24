// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PiecewiseSequence;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::filter::ChunkDecoder;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::PackedChunks;
use super::unpack_chunk_threshold;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::take_chunks::chunked_positions;
use crate::take_chunks::nullable_positions;
use crate::take_chunks::take_in_order;
use crate::take_chunks::valid_positions;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::BitPackingStrategy;
use crate::unpack_iter::BitUnpackedChunks;

const FULL_ARRAY_DECODE_RATIO: usize = 8;
/// Full-decode crossover for chunk-local patched takes (1/16 = 6.25%): above this selectivity,
/// unpacking the full array once and gathering beats per-chunk scratch decode + patching.
const PATCHED_CHUNK_DECODE_RATIO: usize = 16;
/// Full-decode crossover for scattered or non-chunk-aligned patched takes (~0.8%).
const PATCHED_SCATTERED_DECODE_RATIO: usize = 128;
const CHUNK_LOCALITY_SAMPLE_PAIRS: usize = 32;

impl TakeExecute for BitPacked {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        // If the indices are large enough, it's faster to flatten and take the primitive array.
        if indices.len() * FULL_ARRAY_DECODE_RATIO > array.len() {
            let prim = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return prim.into_array().take(indices.clone()).map(Some);
        }

        if let Some(ranges) =
            contiguous_ranges(indices, array.len(), array.dtype().as_ptype(), ctx)?
        {
            let taken_validity = array.validity()?.take(indices)?;
            let ptype = array.dtype().as_ptype();
            let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
                take_ranges::<T>(array, bit_width, &ranges, indices.len(), taken_validity)
            });
            let taken = taken.reinterpret_cast(ptype);
            let taken = if let Some(patches) = array.patches()
                && let Some(patches) = patches.take(indices, ctx)?
            {
                taken.patch(&patches, ctx)?
            } else {
                taken
            };
            return Ok(Some(taken.into_array()));
        }

        // NOTE: we use the unsigned PType because all values in the BitPackedArray must
        //  be non-negative (pre-condition of creating the BitPackedArray).
        let ptype: PType = PType::try_from(array.dtype())?;
        let indices_prim = indices.clone().execute::<PrimitiveArray>(ctx)?;

        if let Some(patches) = array.patches()
            && should_decode_full_for_patches(array, &indices_prim, &patches)?
        {
            let prim = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return prim.into_array().take(indices_prim.into_array()).map(Some);
        }

        let validity = array.validity()?;
        let taken_validity = if indices.is::<Primitive>() {
            validity.take(indices)?
        } else {
            validity.take(&indices_prim.clone().into_array())?
        };

        let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
            match_each_integer_ptype!(indices_prim.ptype(), |I| {
                take_primitive::<T, I>(array, bit_width, &indices_prim, taken_validity, ctx)?
            })
        });
        Ok(Some(taken.into_array()))
    }
}

/// Whether `patches` has chunk offsets aligned with `array`'s FastLanes chunks and already-
/// materialized (or constant) all-valid patch values, so patches can be applied inline per chunk
/// during `take_primitive` without allocating whole patch buffers.
fn can_use_chunk_aligned_patches(
    array: ArrayView<'_, BitPacked>,
    patches: &Patches,
) -> VortexResult<bool> {
    let offset = usize::from(array.offset());
    let num_chunks = (array.len() + offset).div_ceil(FL_CHUNK_SIZE);
    let has_aligned_chunks = patches.indices().is::<Primitive>()
        && (patches.values().is::<Primitive>() || patches.values().is::<Constant>())
        && patches
            .chunk_offsets()
            .as_ref()
            .is_some_and(|chunk_offsets| {
                chunk_offsets.is::<Primitive>()
                    && !chunk_offsets.is_empty()
                    && chunk_offsets.len() >= num_chunks
            })
        && patches.offset_within_chunk().is_some()
        && patches.offset() % FL_CHUNK_SIZE == offset;
    if !has_aligned_chunks {
        return Ok(false);
    }
    Ok(matches!(
        patches.values().validity()?,
        Validity::NonNullable | Validity::AllValid
    ))
}

/// Samples evenly spaced adjacent pairs of `indices` in `O(1)` to detect whether consecutive takes
/// tend to stay within the same 1,024-element FastLanes chunk (sorted/clustered) vs jumping across
/// chunks (random/dispersed).
fn indices_are_chunk_local_typed<I: IntegerPType>(indices: &[I], offset: usize) -> bool {
    if indices.len() <= 1 {
        return true;
    }
    let pairs = indices.len() - 1;
    let samples = pairs.min(CHUNK_LOCALITY_SAMPLE_PAIRS);
    let mut same_chunk = 0usize;
    for sample in 0..samples {
        let pos = sample * pairs / samples;
        let first = AsPrimitive::<usize>::as_(indices[pos]).wrapping_add(offset) / FL_CHUNK_SIZE;
        let second =
            AsPrimitive::<usize>::as_(indices[pos + 1]).wrapping_add(offset) / FL_CHUNK_SIZE;
        if first == second {
            same_chunk += 1;
        }
    }
    same_chunk * 4 >= samples
}

fn indices_are_chunk_local(indices: &PrimitiveArray, offset: usize) -> bool {
    match_each_integer_ptype!(indices.ptype(), |I| {
        indices_are_chunk_local_typed(indices.as_slice::<I>(), offset)
    })
}

/// Decides whether decoding the full patched `BitPacked` array once and gathering from it beats
/// taking from the compressed array.
fn should_decode_full_for_patches(
    array: ArrayView<'_, BitPacked>,
    indices: &PrimitiveArray,
    patches: &Patches,
) -> VortexResult<bool> {
    let array_len = array.len();
    let num_indices = indices.len();
    if array_len == 0 || num_indices == 0 {
        return Ok(false);
    }
    let ratio = if can_use_chunk_aligned_patches(array, patches)?
        && indices_are_chunk_local(indices, usize::from(array.offset()))
    {
        PATCHED_CHUNK_DECODE_RATIO
    } else {
        PATCHED_SCATTERED_DECODE_RATIO
    };
    Ok(num_indices.saturating_mul(ratio) > array_len)
}

fn contiguous_ranges(
    indices: &ArrayRef,
    array_len: usize,
    ptype: PType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Vec<Range<usize>>>> {
    let Some(indices) = indices.as_opt::<PiecewiseSequence>() else {
        return Ok(None);
    };
    if indices
        .multipliers()
        .as_opt::<Constant>()
        .and_then(|multiplier| multiplier.scalar().as_primitive().as_::<u64>())
        != Some(1)
    {
        return Ok(None);
    }
    let sparse_threshold = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
        unpack_chunk_threshold::<T>() as u64
    });

    let starts = indices
        .starts()
        .cast(PType::U64.into())?
        .execute::<PrimitiveArray>(ctx)?;
    let starts = starts.as_slice::<u64>();
    // The primitive path batches short runs that share a packed chunk, so only take the range
    // path when there is at least one run long enough to unpack a whole chunk.
    match indices.lengths().clone().execute::<Columnar>(ctx)? {
        Columnar::Constant(lengths) => {
            let length = lengths
                .scalar()
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("PiecewiseSequence lengths are non-nullable unsigned integers");
            if length <= sparse_threshold {
                return Ok(None);
            }
            collect_ranges(
                starts.iter().map(|&start| (start, length)),
                array_len,
                indices.len(),
            )
            .map(Some)
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths
                .into_array()
                .cast(PType::U64.into())?
                .execute::<PrimitiveArray>(ctx)?;
            let lengths = lengths.as_slice::<u64>();
            if lengths.iter().all(|&length| length <= sparse_threshold) {
                return Ok(None);
            }
            collect_ranges(
                starts.iter().copied().zip_eq(lengths.iter().copied()),
                array_len,
                indices.len(),
            )
            .map(Some)
        }
    }
}

/// Converts `(start, length)` pieces into validated, non-empty ranges.
fn collect_ranges(
    pieces: impl ExactSizeIterator<Item = (u64, u64)>,
    array_len: usize,
    output_len: usize,
) -> VortexResult<Vec<Range<usize>>> {
    let mut ranges = Vec::with_capacity(pieces.len());
    let mut total_len = 0usize;
    for (start, length) in pieces {
        if length == 0 {
            continue;
        }
        let start = usize::try_from(start)?;
        let length = usize::try_from(length)?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        vortex_ensure!(
            end <= array_len,
            "PiecewiseSequenceArray range exceeds array length"
        );
        total_len = total_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
        ranges.push(start..end);
    }
    vortex_ensure_eq!(total_len, output_len);
    Ok(ranges)
}

fn take_ranges<T: NativePType + BitPacking + BitPackedIter<Physical = T>>(
    array: ArrayView<'_, BitPacked>,
    bit_width: u8,
    ranges: &[Range<usize>],
    output_len: usize,
    validity: Validity,
) -> PrimitiveArray {
    let bit_width = usize::from(bit_width);
    let packed = array.packed_slice::<T>();
    let chunks = PackedChunks::new(packed, bit_width);
    let chunk_len = 128 * bit_width / size_of::<T>();
    let offset = usize::from(array.offset());
    let mut output = BufferMut::<T>::with_capacity(output_len);
    let mut scratch = [const { MaybeUninit::uninit() }; FL_CHUNK_SIZE];
    // u64 has the largest sparse decode threshold.
    let mut fragment_indices = [0usize; unpack_chunk_threshold::<u64>()];

    // Vector gathers select runs of components. Long runs are decoded straight into the output so
    // neither the index buffer nor the chunk-grouping work scales with the number of components.
    // Short runs, and the few values a long run selects from a boundary chunk, are cheaper to
    // extract individually. Consecutive short runs are expanded together so that those sharing a
    // chunk are batched.
    let threshold = unpack_chunk_threshold::<T>();
    let mut short_runs_start = 0;
    for (i, range) in ranges.iter().enumerate() {
        if range.len() <= threshold {
            continue;
        }
        take_in_order(
            &chunks,
            &mut output,
            ranges[short_runs_start..i]
                .iter()
                .flat_map(Range::clone)
                .map(|index| index + offset),
            threshold,
            &mut scratch,
        );
        short_runs_start = i + 1;

        let head_len = (FL_CHUNK_SIZE - (range.start + offset) % FL_CHUNK_SIZE) % FL_CHUNK_SIZE;
        let tail_len = (range.end + offset) % FL_CHUNK_SIZE;
        let middle_start = range.start + if head_len <= threshold { head_len } else { 0 };
        let middle_end =
            (range.end - if tail_len <= threshold { tail_len } else { 0 }).max(middle_start);

        unpack_fragment_into(
            &mut output,
            &chunks,
            range.start + offset..middle_start + offset,
            &mut fragment_indices,
        );
        let middle_len = middle_end - middle_start;
        if middle_len != 0 {
            let start = middle_start + offset;
            let first_chunk = start / FL_CHUNK_SIZE;
            let last_chunk = (middle_end + offset).div_ceil(FL_CHUNK_SIZE);
            let mut chunks = BitUnpackedChunks::try_new_with_strategy(
                BitPackingStrategy,
                &packed[first_chunk * chunk_len..last_chunk * chunk_len],
                bit_width,
                start % FL_CHUNK_SIZE,
                middle_len,
                &mut scratch,
            )
            .vortex_expect("validated range lies within the packed buffer");
            let output_start = output.len();
            chunks.decode_into(&mut output.spare_capacity_mut()[..middle_len]);
            // SAFETY: `decode_into` initialized the `middle_len` values following `output_start`.
            unsafe { output.set_len(output_start + middle_len) };
        }
        unpack_fragment_into(
            &mut output,
            &chunks,
            middle_end + offset..range.end + offset,
            &mut fragment_indices,
        );
    }
    take_in_order(
        &chunks,
        &mut output,
        ranges[short_runs_start..]
            .iter()
            .flat_map(Range::clone)
            .map(|index| index + offset),
        threshold,
        &mut scratch,
    );
    PrimitiveArray::new(output, validity)
}

/// Appends the values at `positions`, which lie within one chunk, without decoding the whole chunk.
fn unpack_fragment_into<T: NativePType + BitPacking>(
    output: &mut BufferMut<T>,
    chunks: &PackedChunks<'_, T>,
    positions: Range<usize>,
    indices: &mut [usize],
) {
    if positions.is_empty() {
        return;
    }
    let chunk = positions.start / FL_CHUNK_SIZE;
    let indices = &mut indices[..positions.len()];
    for (index, position) in indices.iter_mut().zip(positions) {
        *index = position % FL_CHUNK_SIZE;
    }
    let start = output.len();
    chunks.decode_indices(
        chunk,
        indices,
        &mut output.spare_capacity_mut()[..indices.len()],
    );
    // SAFETY: `decode_indices` wrote one value for each index.
    unsafe { output.set_len(start + indices.len()) };
}

/// Borrowed unsigned patch indices, avoiding multiplying generic instantiations by patch index
/// width across all `(value_type, take_index_type)` pairs.
#[derive(Clone, Copy)]
enum PatchIndicesSlice<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl<'a> PatchIndicesSlice<'a> {
    fn from_primitive(array: &'a PrimitiveArray) -> Self {
        match array.ptype() {
            PType::U8 => Self::U8(array.as_slice::<u8>()),
            PType::U16 => Self::U16(array.as_slice::<u16>()),
            PType::U32 => Self::U32(array.as_slice::<u32>()),
            PType::U64 => Self::U64(array.as_slice::<u64>()),
            other => unreachable!("patch indices must be unsigned integers, got {other}"),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::U8(slice) => slice.len(),
            Self::U16(slice) => slice.len(),
            Self::U32(slice) => slice.len(),
            Self::U64(slice) => slice.len(),
        }
    }

    #[inline]
    fn get(&self, idx: usize) -> usize {
        match self {
            Self::U8(slice) => slice[idx].as_(),
            Self::U16(slice) => slice[idx].as_(),
            Self::U32(slice) => slice[idx].as_(),
            Self::U64(slice) => slice[idx].as_(),
        }
    }
}

#[derive(Clone, Copy)]
enum PatchValues<'a, T> {
    Slice(&'a [T]),
    Constant(T),
}

impl<T: Copy> PatchValues<'_, T> {
    #[inline]
    fn slice(&self, range: Range<usize>) -> Self {
        match self {
            Self::Slice(values) => Self::Slice(&values[range]),
            Self::Constant(value) => Self::Constant(*value),
        }
    }

    #[inline]
    fn get(&self, idx: usize) -> T {
        match self {
            Self::Slice(values) => values[idx],
            Self::Constant(value) => *value,
        }
    }
}

struct ChunkPatches<'a, T> {
    indices: PatchIndicesSlice<'a>,
    values: PatchValues<'a, T>,
    chunk_offsets: PatchIndicesSlice<'a>,
    base_chunk_offset: usize,
    base_patch_offset: usize,
    offset_within_chunk: usize,
}

impl<T: Copy> ChunkPatches<'_, T> {
    #[inline]
    fn chunk_range(&self, chunk: usize) -> Range<usize> {
        let num_patches = self.indices.len();
        let start = self
            .chunk_offsets
            .get(chunk)
            .saturating_sub(self.base_patch_offset)
            .saturating_sub(self.offset_within_chunk)
            .min(num_patches);
        let end = if chunk + 1 < self.chunk_offsets.len() {
            self.chunk_offsets
                .get(chunk + 1)
                .saturating_sub(self.base_patch_offset)
                .saturating_sub(self.offset_within_chunk)
                .min(num_patches)
        } else {
            num_patches
        };
        start..end.max(start)
    }

    #[inline]
    fn patch_scratch_range(
        &self,
        chunk: usize,
        range: Range<usize>,
        scratch: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) {
        let patch_chunk_base = self.base_chunk_offset + chunk * FL_CHUNK_SIZE;
        let chunk_values = self.values.slice(range.clone());
        match self.indices {
            PatchIndicesSlice::U8(indices) => {
                apply_chunk_patches_typed(scratch, &indices[range], chunk_values, patch_chunk_base)
            }
            PatchIndicesSlice::U16(indices) => {
                apply_chunk_patches_typed(scratch, &indices[range], chunk_values, patch_chunk_base)
            }
            PatchIndicesSlice::U32(indices) => {
                apply_chunk_patches_typed(scratch, &indices[range], chunk_values, patch_chunk_base)
            }
            PatchIndicesSlice::U64(indices) => {
                apply_chunk_patches_typed(scratch, &indices[range], chunk_values, patch_chunk_base)
            }
        }
    }

    #[inline]
    fn patch_sparse_range(
        &self,
        chunk: usize,
        range: Range<usize>,
        within: &[usize],
        dst: &mut [MaybeUninit<T>],
    ) {
        let patch_chunk_base = self.base_chunk_offset + chunk * FL_CHUNK_SIZE;
        let chunk_values = self.values.slice(range.clone());
        match self.indices {
            PatchIndicesSlice::U8(indices) => patch_sparse_within_typed(
                dst,
                within,
                &indices[range],
                chunk_values,
                patch_chunk_base,
            ),
            PatchIndicesSlice::U16(indices) => patch_sparse_within_typed(
                dst,
                within,
                &indices[range],
                chunk_values,
                patch_chunk_base,
            ),
            PatchIndicesSlice::U32(indices) => patch_sparse_within_typed(
                dst,
                within,
                &indices[range],
                chunk_values,
                patch_chunk_base,
            ),
            PatchIndicesSlice::U64(indices) => patch_sparse_within_typed(
                dst,
                within,
                &indices[range],
                chunk_values,
                patch_chunk_base,
            ),
        }
    }
}

fn apply_chunk_patches_typed<T: Copy, P: UnsignedPType>(
    scratch: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    chunk_patch_indices: &[P],
    chunk_values: PatchValues<'_, T>,
    patch_chunk_base: usize,
) {
    match chunk_values {
        PatchValues::Slice(values) => {
            for (&patch_index, &patch_value) in chunk_patch_indices.iter().zip(values) {
                scratch[patch_index.as_() - patch_chunk_base].write(patch_value);
            }
        }
        PatchValues::Constant(patch_value) => {
            for &patch_index in chunk_patch_indices {
                scratch[patch_index.as_() - patch_chunk_base].write(patch_value);
            }
        }
    }
}

fn patch_sparse_within_typed<T: Copy, P: UnsignedPType>(
    dst: &mut [MaybeUninit<T>],
    within: &[usize],
    chunk_patch_indices: &[P],
    chunk_values: PatchValues<'_, T>,
    patch_chunk_base: usize,
) {
    // Invariant: every element in `chunk_patch_indices[..lo]` is strictly less than
    // `patch_chunk_base + prev`. On sorted `within` this advances monotonically in O(1) per miss,
    // while unsorted inputs reset `lo` and still binary-search within the L1-resident chunk slice.
    let mut prev = 0usize;
    let mut lo = 0usize;
    for (slot, &pos_in_chunk) in dst.iter_mut().zip(within) {
        if pos_in_chunk < prev {
            lo = 0;
        }
        prev = pos_in_chunk;
        if lo >= chunk_patch_indices.len() {
            continue;
        }
        let Some(needle) = <P as num_traits::NumCast>::from(patch_chunk_base + pos_in_chunk) else {
            continue;
        };
        let first = chunk_patch_indices[lo];
        if needle < first {
            continue;
        }
        if needle == first {
            slot.write(chunk_values.get(lo));
            continue;
        }
        let tail = &chunk_patch_indices[lo + 1..];
        match tail.binary_search(&needle) {
            Ok(rel) => {
                let matched = lo + 1 + rel;
                slot.write(chunk_values.get(matched));
                lo = matched;
            }
            Err(rel) => {
                lo = lo + 1 + rel;
            }
        }
    }
}

#[inline]
fn take_in_order_patched<T: Copy>(
    decoder: &impl ChunkDecoder<T>,
    output: &mut BufferMut<T>,
    positions: impl Iterator<Item = usize>,
    sparse_limit: usize,
    scratch: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    patches: &ChunkPatches<'_, T>,
) {
    chunked_positions(positions, |chunk, within| {
        let patch_range = patches.chunk_range(chunk);
        let start = output.len();
        let dst = &mut output.spare_capacity_mut()[..within.len()];
        if within.len() > sparse_limit {
            decoder.decode_chunk(chunk, scratch);
            if !patch_range.is_empty() {
                patches.patch_scratch_range(chunk, patch_range, scratch);
            }
            for (slot, &index) in dst.iter_mut().zip(within) {
                // SAFETY: every position in `within` is below `FL_CHUNK_SIZE`, and `decode_chunk`
                // initialized every slot of `scratch`.
                slot.write(unsafe { scratch.get_unchecked(index).assume_init() });
            }
        } else {
            decoder.decode_indices(chunk, within, dst);
            if !patch_range.is_empty() {
                patches.patch_sparse_range(chunk, patch_range, within, dst);
            }
        }
        // SAFETY: every slot in `dst` (`within.len()` elements) was initialized above.
        unsafe { output.set_len(start + within.len()) };
    });
}

fn take_primitive<T: NativePType + BitPacking, I: IntegerPType>(
    array: ArrayView<'_, BitPacked>,
    bit_width: u8,
    indices: &PrimitiveArray,
    taken_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    if indices.is_empty() {
        return Ok(PrimitiveArray::new(Buffer::<T>::empty(), taken_validity)
            .reinterpret_cast(array.dtype().as_ptype()));
    }

    let index_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    // Like the rest of BitPacked's take, this trusts the caller for the bounds of valid indices: a
    // check costs as much as the take for small batches of `u64` indices, which the baseline
    // instruction set cannot compare in vectors.
    let indices_slice = indices.as_slice::<I>();
    let chunks = PackedChunks::new(array.packed_slice::<T>(), usize::from(bit_width));
    let offset = usize::from(array.offset());
    let sparse_limit = unpack_chunk_threshold::<T>();
    let mut output = BufferMut::<T>::with_capacity(indices.len());
    let mut scratch = [const { MaybeUninit::uninit() }; FL_CHUNK_SIZE];

    if let Some(patches) = array.patches()
        && can_use_chunk_aligned_patches(array, &patches)?
    {
        let patch_indices_prim = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let chunk_offsets_prim = patches
            .chunk_offsets()
            .as_ref()
            .vortex_expect("checked chunk_offsets is Some")
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let chunk_offsets = PatchIndicesSlice::from_primitive(&chunk_offsets_prim);
        let patch_values_prim;
        let patch_values = if let Some(constant) = patches.values().as_opt::<Constant>() {
            let value = constant
                .scalar()
                .as_primitive()
                .pvalue()
                .vortex_expect("checked patch values are non-null")
                .reinterpret_cast(T::PTYPE)
                .cast::<T>()?;
            PatchValues::Constant(value)
        } else {
            patch_values_prim = patches
                .values()
                .clone()
                .execute::<PrimitiveArray>(ctx)?
                .reinterpret_cast(T::PTYPE);
            PatchValues::Slice(patch_values_prim.as_slice::<T>())
        };
        let chunk_patches = ChunkPatches {
            indices: PatchIndicesSlice::from_primitive(&patch_indices_prim),
            values: patch_values,
            chunk_offsets,
            base_chunk_offset: patches.offset() - offset,
            base_patch_offset: chunk_offsets.get(0),
            offset_within_chunk: patches
                .offset_within_chunk()
                .vortex_expect("checked offset_within_chunk is Some"),
        };

        match index_validity {
            Mask::AllTrue(_) => take_in_order_patched(
                &chunks,
                &mut output,
                valid_positions(indices_slice, offset),
                sparse_limit,
                &mut scratch,
                &chunk_patches,
            ),
            _ => take_in_order_patched(
                &chunks,
                &mut output,
                nullable_positions(indices_slice, &index_validity, offset),
                sparse_limit,
                &mut scratch,
                &chunk_patches,
            ),
        }

        return Ok(
            PrimitiveArray::new(output, taken_validity).reinterpret_cast(array.dtype().as_ptype())
        );
    }

    match index_validity {
        Mask::AllTrue(_) => take_in_order(
            &chunks,
            &mut output,
            valid_positions(indices_slice, offset),
            sparse_limit,
            &mut scratch,
        ),
        _ => take_in_order(
            &chunks,
            &mut output,
            nullable_positions(indices_slice, &index_validity, offset),
            sparse_limit,
            &mut scratch,
        ),
    }

    let unpatched_taken =
        PrimitiveArray::new(output, taken_validity).reinterpret_cast(array.dtype().as_ptype());
    if let Some(patches) = array.patches()
        && let Some(patches) = patches.take(&indices.clone().into_array(), ctx)?
    {
        return unpatched_taken.patch(&patches, ctx);
    }

    Ok(unpatched_taken)
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::distr::Uniform;
    use rand::rng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PiecewiseSequenceArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::TakeExecute;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::builder_with_capacity_in;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::patches::Patches;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedData;
    use crate::FL_CHUNK_SIZE;
    use crate::bitpacking::array::BitPackedArrayExt;
    use crate::bitpacking::compute::take::should_decode_full_for_patches;
    use crate::bitpacking::compute::take::take_primitive;
    use crate::bitpacking::compute::take::unpack_chunk_threshold;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn sparse_extraction_covers_batch_and_full_chunk_paths() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        macro_rules! check_type {
            ($T:ty, $bit_width:expr) => {{
                let values = (0..2_048)
                    .map(|index| (index % 127) as $T)
                    .collect::<Vec<_>>();
                let packed = BitPackedData::encode(
                    &PrimitiveArray::from_iter(values.iter().copied()).into_array(),
                    $bit_width,
                    &mut ctx,
                )?;
                let threshold = unpack_chunk_threshold::<$T>();

                for selected in [threshold, threshold + 1] {
                    let indices = (0..selected)
                        .map(|index| (index * 1_024 / selected) as u32)
                        .collect::<Vec<_>>();
                    let actual = take_primitive::<$T, u32>(
                        packed.as_view(),
                        $bit_width,
                        &PrimitiveArray::from_iter(indices.iter().copied()),
                        Validity::NonNullable,
                        &mut ctx,
                    )?;
                    let expected = indices
                        .iter()
                        .map(|&index| values[index as usize])
                        .collect::<Vec<_>>();
                    assert_eq!(actual.as_slice::<$T>(), expected);
                }
            }};
        }

        check_type!(u8, 7);
        check_type!(u16, 15);
        check_type!(u32, 31);
        check_type!(u64, 63);
        Ok(())
    }

    #[test]
    fn sparse_take_preserves_nullable_order_and_duplicates() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..8_192).map(|index| index as u32).collect::<Vec<_>>();
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter(values.iter().copied()).into_array(),
            13,
            &mut ctx,
        )?;
        let indices = [
            Some(3_073u32),
            Some(2),
            None,
            Some(2),
            Some(1_025),
            Some(3_073),
            Some(0),
        ];
        let actual = packed
            .take(PrimitiveArray::from_option_iter(indices).into_array())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected = PrimitiveArray::from_option_iter(
            indices.map(|index| index.map(|index| values[index as usize])),
        );
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn take_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let indices = buffer![0, 125, 2047, 2049, 2151, 2790].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();

        let primitive_result = bitpacked.take(indices).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18]),
            &mut ctx
        );
    }

    #[test]
    fn take_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let unpacked = Buffer::from_iter(0u32..1024).into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 2, &mut ctx).unwrap();

        let indices = buffer![0, 2, 4, 6].into_array();

        let primitive_result = bitpacked.take(indices).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u32, 2, 4, 6]),
            &mut ctx
        );
    }

    #[test]
    fn take_sliced_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let indices = buffer![1919, 1921].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let primitive_result = sliced.take(indices).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([31u8, 33]),
            &mut ctx
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn take_random_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let num_patches: usize = 128;
        let values = (0..u16::MAX as u32 + num_patches as u32).collect::<Buffer<_>>();
        let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let packed = BitPackedData::encode(&uncompressed.into_array(), 16, &mut ctx).unwrap();
        assert!(packed.patches().is_some());

        let rng = rng();
        let range = Uniform::new(0, values.len()).unwrap();
        let random_indices =
            PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));
        let taken = packed.take(random_indices.clone().into_array()).unwrap();

        // sanity check
        random_indices
            .as_slice::<u32>()
            .iter()
            .enumerate()
            .for_each(|(ti, i)| {
                assert_eq!(
                    u32::try_from(&packed.execute_scalar(*i as usize, &mut ctx).unwrap()).unwrap(),
                    values[*i as usize]
                );
                assert_eq!(
                    u32::try_from(&taken.execute_scalar(ti, &mut ctx).unwrap()).unwrap(),
                    values[*i as usize]
                );
            });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn take_signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let start =
            BitPackedData::encode(&buffer![1i32, 2i32, 3i32, 4i32].into_array(), 1, &mut ctx)
                .unwrap();

        let taken_primitive = take_primitive::<u32, u64>(
            start.as_view(),
            1,
            &PrimitiveArray::from_iter([0u64, 1, 2, 3]),
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap();
        assert_arrays_eq!(
            taken_primitive,
            PrimitiveArray::from_iter([1i32, 2, 3, 4]),
            &mut ctx
        );
    }

    #[test]
    fn take_nullable_with_nullables() {
        let mut ctx = SESSION.create_execution_ctx();
        let start =
            BitPackedData::encode(&buffer![1i32, 2i32, 3i32, 4i32].into_array(), 1, &mut ctx)
                .unwrap();

        let taken_primitive = start
            .take(
                PrimitiveArray::from_option_iter([Some(0u64), Some(1), None, Some(3)]).into_array(),
            )
            .unwrap();
        assert_arrays_eq!(
            taken_primitive,
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4)]),
            &mut ctx
        );
        let taken_primitive_prim = taken_primitive.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_eq!(taken_primitive_prim.invalid_count(&mut ctx).unwrap(), 1);
    }

    #[rstest]
    #[case(0, 0, false, false, u64::MAX, PType::U64)]
    #[case(7, 0, false, false, u64::MAX, PType::U8)]
    #[case(7, 137, true, true, 0, PType::I32)]
    #[case(15, 1023, false, true, 0, PType::U16)]
    #[case(4, 137, false, true, 0, PType::U8)]
    #[case(4, 1023, true, true, 0, PType::U8)]
    #[case(63, 1023, false, true, 0, PType::U64)]
    #[case(62, 137, true, true, 0, PType::I64)]
    fn test_take_piecewise_ranges(
        #[case] bit_width: u8,
        #[case] slice_offset: usize,
        #[case] patched: bool,
        #[case] nullable: bool,
        #[case] empty_start: u64,
        #[case] ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_len: usize = 65523;
        let values = PrimitiveArray::from_option_iter((0..source_len).map(|i| {
            if nullable && i.is_multiple_of(29) {
                None
            } else if patched && i.is_multiple_of(31) {
                Some(1u64 << bit_width)
            } else {
                Some((i as u64).wrapping_mul(0x0123_4567_89ab_cdef) & ((1u64 << bit_width) - 1))
            }
        }))
        .into_array()
        .cast(DType::Primitive(ptype, Nullability::Nullable))?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
        let packed = BitPackedData::encode(&values, bit_width, &mut ctx)?;
        let packed =
            BitPacked::slice(packed.as_view(), slice_offset..source_len, &mut ctx)?.unwrap();
        let values = values.slice(slice_offset..source_len)?;
        let indices = PiecewiseSequenceArray::try_new(
            buffer![
                empty_start,
                17,
                991,
                1000,
                3072,
                12000,
                17000,
                17,
                (source_len - slice_offset - 257) as u64
            ]
            .into_array(),
            buffer![0u16, 256, 257, 511, 768, 1024, 1536, 256, 257].into_array(),
            ConstantArray::new(1u8, 9).into_array(),
            4865,
        )?
        .into_array();
        let expanded = indices
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        let actual = BitPacked::take(packed.as_::<BitPacked>(), &indices, &mut ctx)?.unwrap();
        assert_arrays_eq!(actual, values.take(expanded)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_piecewise_fallbacks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0u32..32768).map(|i| i % 127)).into_array();
        let packed = BitPackedData::encode(&values, 7, &mut ctx)?;
        for (length, multiplier, constant) in [
            (7u8, 1u8, true),
            (65, 0, true),
            (65, 2, true),
            (65, 1, false),
        ] {
            let multipliers = if constant {
                ConstantArray::new(multiplier, 2).into_array()
            } else {
                buffer![multiplier, multiplier].into_array()
            };
            for (lengths, len) in [
                (buffer![length, 128].into_array(), usize::from(length) + 128),
                (
                    ConstantArray::new(length, 2).into_array(),
                    usize::from(length) * 2,
                ),
            ] {
                let indices = PiecewiseSequenceArray::try_new(
                    buffer![1017u16, 1023].into_array(),
                    lengths,
                    multipliers.clone(),
                    len,
                )?
                .into_array();
                let expanded = indices
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?
                    .into_array();
                assert_arrays_eq!(packed.take(indices)?, values.take(expanded)?, &mut ctx);
            }
        }
        Ok(())
    }

    #[rstest]
    fn test_take_piecewise_mixed_run_lengths(
        #[values(0, 137)] slice_offset: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0u32..16384).map(|i| i % 127)).into_array();
        let packed = BitPackedData::encode(&values, 7, &mut ctx)?;
        let packed = BitPacked::slice(packed.as_view(), slice_offset..16384, &mut ctx)?.unwrap();
        let values = values.slice(slice_offset..16384)?;
        // Short runs sharing a chunk sit between long runs that cross chunk boundaries. The run at
        // 1900 leaves a short tail in the chunk that the following short run also selects from.
        let indices = PiecewiseSequenceArray::try_new(
            buffer![10u32, 20, 1000, 30, 40, 4000, 50, 4090, 1900, 2070].into_array(),
            buffer![3u32, 5, 300, 2, 0, 2048, 1, 6, 160, 3].into_array(),
            ConstantArray::new(1u8, 10).into_array(),
            2528,
        )?
        .into_array();
        let expanded = indices
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        let actual = BitPacked::take(packed.as_::<BitPacked>(), &indices, &mut ctx)?.unwrap();
        assert_arrays_eq!(actual, values.take(expanded)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(PType::U8, 4, 16)]
    #[case(PType::U16, 9, 32)]
    #[case(PType::U32, 16, 64)]
    #[case(PType::U64, 33, 160)]
    fn test_take_piecewise_sparse_threshold(
        #[case] ptype: PType,
        #[case] bit_width: u8,
        #[case] threshold: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0u64..8192).map(|i| i.wrapping_mul(0x0123_4567_89ab_cdef) & ((1u64 << bit_width) - 1)),
        )
        .into_array()
        .cast(ptype.into())?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
        let packed = BitPackedData::encode(&values, bit_width, &mut ctx)?;
        for length in [threshold, threshold + 1, 2 * threshold + 1] {
            // A run split across two chunks can require sparse extraction on both sides.
            let indices = PiecewiseSequenceArray::try_new(
                buffer![0u64, (1024 - threshold / 2) as u64, 4096].into_array(),
                ConstantArray::new(length as u64, 3).into_array(),
                ConstantArray::new(1u64, 3).into_array(),
                length * 3,
            )?
            .into_array();
            let expanded = indices
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?
                .into_array();
            assert_arrays_eq!(packed.take(indices)?, values.take(expanded)?, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    fn test_take_piecewise_fsl_integration(
        #[values(false, true)] blocked: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter(
            (0usize..65536).map(|i| (!i.is_multiple_of(29)).then_some((i % 128) as u8)),
        )
        .into_array();
        let packed = if blocked {
            let widths = (0..64)
                .map(|i| if i % 2 == 0 { 4 } else { 7 })
                .collect::<Vec<_>>();
            BitPackedData::encode_blocked(&values, &widths, &mut ctx)?
        } else {
            BitPackedData::encode(&values, 4, &mut ctx)?
        };
        let validity = Validity::from_iter((0usize..1024).map(|i| !i.is_multiple_of(11)));
        let expected = FixedSizeListArray::new(values, 64, validity.clone(), 1024);
        let array = FixedSizeListArray::new(packed.into_array(), 64, validity, 1024);
        let indices =
            PrimitiveArray::from_option_iter([Some(1000u32), None, Some(0), Some(16), Some(16)])
                .into_array();
        assert_arrays_eq!(
            array.take(indices.clone())?,
            expected.take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    /// A lazy take of runs, as a fixed-size-list take makes, is appended to a builder through the
    /// take kernel with the runs, not with the runs expanded into one index per value.
    #[rstest]
    fn test_append_lazy_piecewise_take(#[values(0, 1000)] first_start: u64) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0u32..65536).map(|i| i % 1000)).into_array();
        let packed = BitPackedData::encode(&values, 10, &mut ctx)?.into_array();

        let indices = PiecewiseSequenceArray::try_new(
            buffer![first_start, 30720, 2048].into_array(),
            ConstantArray::new(1024u64, 3).into_array(),
            ConstantArray::new(1u64, 3).into_array(),
            3072,
        )?
        .into_array();
        let lazy = DictArray::try_new(indices.clone(), packed)?.into_array();

        let mut builder = builder_with_capacity_in(lazy.dtype(), lazy.len(), ctx.allocator());
        lazy.append_to_builder(builder.as_mut(), &mut ctx)?;

        let expanded = indices.execute::<PrimitiveArray>(&mut ctx)?.into_array();
        assert_arrays_eq!(builder.finish(), values.take(expanded)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(32700, 128, 128)]
    #[case(1, 128, 127)]
    #[case(u64::MAX, 128, 128)]
    fn test_take_piecewise_invalid_range(
        #[case] start: u64,
        #[case] length: u64,
        #[case] declared_len: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..32768).map(|i| i as u32 % 127)).into_array();
        let packed = BitPackedData::encode(&values, 7, &mut ctx)?;
        let indices = PiecewiseSequenceArray::try_new(
            buffer![start].into_array(),
            buffer![length].into_array(),
            ConstantArray::new(1u64, 1).into_array(),
            declared_len,
        )?
        .into_array();
        assert!(
            packed
                .take(indices)?
                .execute::<PrimitiveArray>(&mut ctx)
                .is_err()
        );
        Ok(())
    }

    fn bp(array: vortex_array::ArrayRef, bit_width: u8) -> BitPackedArray {
        BitPackedData::encode(&array, bit_width, &mut SESSION.create_execution_ctx()).unwrap()
    }

    #[rstest]
    #[case(bp(PrimitiveArray::from_iter((0..100).map(|i| (i % 63) as u8)).into_array(), 6))]
    #[case(bp(PrimitiveArray::from_iter((0..256).map(|i| i as u32)).into_array(), 8))]
    #[case(bp(buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array(), 3))]
    #[case(bp(
        PrimitiveArray::from_option_iter([Some(10u16), None, Some(20), Some(30), None]).into_array(),
        5
    ))]
    #[case(bp(buffer![42u32].into_array(), 6))]
    #[case(bp(PrimitiveArray::from_iter((0..1024).map(|i| i as u32)).into_array(), 8))]
    fn test_take_bitpacked_conformance(#[case] bitpacked: BitPackedArray) {
        test_take_conformance(&bitpacked.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case(PType::U8, 4, 0, false, false)]
    #[case(PType::I16, 6, 137, true, true)]
    #[case(PType::U32, 10, 1161, true, true)]
    #[case(PType::I64, 10, 137, false, true)]
    fn test_take_heavy_patched_sorted_and_random(
        #[case] ptype: PType,
        #[case] bit_width: u8,
        #[case] slice_offset: usize,
        #[case] nullable_values: bool,
        #[case] nullable_indices: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_len = 24_576usize;
        let mask = (1u64 << bit_width) - 1;
        let patch_val = 1u64 << bit_width;
        let base_nullability = if nullable_values {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };

        let raw_values = PrimitiveArray::from_option_iter((0..source_len).map(|idx| {
            if nullable_values && idx.is_multiple_of(29) {
                None
            } else if idx.is_multiple_of(61) {
                Some(patch_val + (idx as u64 % 31))
            } else {
                Some((idx as u64).wrapping_mul(13) & mask)
            }
        }))
        .into_array()
        .cast(DType::Primitive(ptype, base_nullability))?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();

        let packed = BitPackedData::encode(&raw_values, bit_width, &mut ctx)?;
        assert!(packed.patches().is_some());
        let packed = if slice_offset > 0 {
            BitPacked::slice(packed.as_view(), slice_offset..source_len, &mut ctx)?
                .expect("BitPacked slice")
        } else {
            packed.into_array()
        };
        let expected_source = raw_values.slice(slice_offset..source_len)?;
        let len = expected_source.len();

        let mut rng = StdRng::seed_from_u64(42);

        for (selectivity_pct, sorted, expect_full_decode) in
            [(1usize, true, false), (10, true, true), (10, false, true)]
        {
            let count = len * selectivity_pct / 100;
            let mut raw_indices: Vec<u32> = (0..count)
                .map(|_| rng.random_range(0..len as u32))
                .collect();
            if sorted {
                raw_indices.sort_unstable();
            }
            let indices = if nullable_indices {
                PrimitiveArray::from_option_iter(raw_indices.into_iter().enumerate().map(
                    |(pos, value)| {
                        if pos.is_multiple_of(19) {
                            None
                        } else {
                            Some(value)
                        }
                    },
                ))
                .into_array()
            } else {
                PrimitiveArray::from_iter(raw_indices).into_array()
            };

            let indices_prim = indices.clone().execute::<PrimitiveArray>(&mut ctx)?;
            let bp_view = packed.as_::<BitPacked>();
            let patches = bp_view.patches().expect("sliced array retains patches");
            assert_eq!(
                should_decode_full_for_patches(bp_view, &indices_prim, &patches)?,
                expect_full_decode
            );

            let actual = packed.take(indices.clone())?;
            let expected = expected_source.take(indices)?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    #[case(0, false)]
    #[case(137, true)]
    fn test_take_patched_without_chunk_offsets(
        #[case] slice_offset: usize,
        #[case] nullable_indices: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_len = 24_576usize;
        let values = PrimitiveArray::from_iter((0..source_len).map(|idx| {
            if idx.is_multiple_of(53) {
                (500 + (idx % 500)) as i16
            } else {
                (idx % 64) as i16
            }
        }))
        .into_array();
        let encoded = BitPackedData::encode(&values, 6, &mut ctx)?;
        let sliced = if slice_offset > 0 {
            BitPacked::slice(encoded.as_view(), slice_offset..source_len, &mut ctx)?
                .expect("BitPacked slice")
                .as_::<BitPacked>()
                .into_owned()
        } else {
            encoded
        };
        let parts = BitPacked::into_parts(sliced);
        let orig_patches = parts.patches.expect("expected patches");
        let patches_without_offsets = Patches::new(
            orig_patches.array_len(),
            orig_patches.offset(),
            orig_patches.indices().clone(),
            orig_patches.values().clone(),
            None,
            &mut ctx,
        )?;
        let bitpacked = BitPacked::try_new(
            parts.packed,
            PType::I16,
            parts.validity,
            Some(patches_without_offsets),
            6,
            parts.len,
            parts.offset,
        )?
        .into_array();
        let expected_source = values.slice(slice_offset..source_len)?;
        let len = expected_source.len();

        let mut rng = StdRng::seed_from_u64(7);
        for (count, expect_full_decode) in [(8usize, false), (len / 10, true)] {
            let mut raw: Vec<u32> = (0..count)
                .map(|_| rng.random_range(0..len as u32))
                .collect();
            raw.sort_unstable();
            let indices = if nullable_indices {
                PrimitiveArray::from_option_iter(raw.into_iter().enumerate().map(|(pos, value)| {
                    if pos.is_multiple_of(7) {
                        None
                    } else {
                        Some(value)
                    }
                }))
                .into_array()
            } else {
                PrimitiveArray::from_iter(raw).into_array()
            };

            let bp_view = bitpacked.as_::<BitPacked>();
            let indices_prim = indices.clone().execute::<PrimitiveArray>(&mut ctx)?;
            let patches = bp_view.patches().expect("patches present");
            assert_eq!(
                should_decode_full_for_patches(bp_view, &indices_prim, &patches)?,
                expect_full_decode
            );
            assert_arrays_eq!(
                bitpacked.take(indices.clone())?,
                expected_source.take(indices)?,
                &mut ctx
            );
        }
        Ok(())
    }

    #[test]
    fn test_take_patched_fully_patched_chunk_and_nested_slices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_len = 6 * FL_CHUNK_SIZE;
        // Chunk 1 is fully patched; other chunks have sparse patches extending past mid-chunk ends.
        let values = PrimitiveArray::from_iter((0..source_len).map(|idx| {
            if (FL_CHUNK_SIZE..2 * FL_CHUNK_SIZE).contains(&idx) || idx.is_multiple_of(17) {
                (500 + (idx % 1000)) as i16
            } else {
                (idx % 64) as i16
            }
        }))
        .into_array();
        let encoded = BitPackedData::encode(&values, 6, &mut ctx)?;

        // Slice ending mid-chunk with patches past its end, then slice again so
        // `offset_within_chunk` accumulates across slices.
        let first_slice =
            BitPacked::slice(encoded.as_view(), 137..4 * FL_CHUNK_SIZE + 500, &mut ctx)?
                .expect("first BitPacked slice");
        let nested_slice = BitPacked::slice(
            first_slice.as_::<BitPacked>(),
            200..3 * FL_CHUNK_SIZE + 100,
            &mut ctx,
        )?
        .expect("nested BitPacked slice");
        let expected_nested = values.slice(337..337 + (3 * FL_CHUNK_SIZE - 100))?;

        // Exercise both sparse-in-chunk (<= 32) and dense-in-chunk (> 32) selections on the
        // nested slice and on the fully patched chunk.
        let sparse_indices =
            PrimitiveArray::from_iter([0u32, 1, 10, 700, 750, 1_500, 2_000, 2_500, 2_900])
                .into_array();
        let dense_indices =
            PrimitiveArray::from_iter((680u32..740).chain(1_600u32..1_660).chain(2_700u32..2_760))
                .into_array();

        for indices in [sparse_indices, dense_indices] {
            assert_arrays_eq!(
                nested_slice.take(indices.clone())?,
                expected_nested.take(indices)?,
                &mut ctx
            );
        }
        Ok(())
    }

    #[test]
    fn test_take_patched_duplicates_out_of_order_and_zero_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 8_192usize;
        for bit_width in [0u8, 5u8] {
            let values = PrimitiveArray::from_iter((0..len).map(|idx| {
                if idx.is_multiple_of(37) {
                    (900 + idx % 100) as i16
                } else if bit_width == 0 {
                    0i16
                } else {
                    (idx % (1 << bit_width)) as i16
                }
            }))
            .into_array();
            let packed = BitPackedData::encode(&values, bit_width, &mut ctx)?.into_array();

            // Repeated patched indices, repeated unpatched indices, descending within chunk,
            // and chunk-boundary indices.
            let tricky = PrimitiveArray::from_iter([
                0u32, 0, 37, 37, 37, 36, 36, 38, 38, 1023, 74, 0, 1024, 1024, 1036, 1036, 2047,
                2048, 2072, 2072, 100, 100, 74, 37,
            ])
            .into_array();
            assert_arrays_eq!(packed.take(tricky.clone())?, values.take(tricky)?, &mut ctx);

            // All-null take indices over a patched array.
            let all_null =
                PrimitiveArray::from_option_iter(vec![Option::<u32>::None; 128]).into_array();
            assert_arrays_eq!(
                packed.take(all_null.clone())?,
                values.take(all_null)?,
                &mut ctx
            );
        }
        Ok(())
    }

    #[test]
    fn test_take_patched_all_index_widths() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // Small array so u8 / i8 indices stay in bounds.
        let small_values = PrimitiveArray::from_iter((0..120u16).map(|idx| {
            if idx.is_multiple_of(7) {
                500 + idx
            } else {
                idx % 32
            }
        }))
        .into_array();
        let small_packed = BitPackedData::encode(&small_values, 5, &mut ctx)?;
        for index_ptype in [
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
        ] {
            // 6 indices out of 120 (5% < 6.25% chunk-local cutoff) so inline patch lookup runs.
            let indices_prim = PrimitiveArray::from_iter([0u8, 7, 7, 14, 70, 119])
                .into_array()
                .cast(index_ptype.into())?
                .execute::<PrimitiveArray>(&mut ctx)?;
            let patches = small_packed.patches().expect("small array has patches");
            assert!(!should_decode_full_for_patches(
                small_packed.as_view(),
                &indices_prim,
                &patches,
            )?);
            let indices = indices_prim.into_array();
            assert_arrays_eq!(
                small_packed.clone().into_array().take(indices.clone())?,
                small_values.take(indices)?,
                &mut ctx
            );
        }

        // Larger array exercising inline chunk-patched (3.125% sorted) and full-decode (3.125%
        // random) paths across u16..u64 and i16..i64.
        let large_len = 16_384usize;
        let large_values = PrimitiveArray::from_iter((0..large_len).map(|idx| {
            if idx.is_multiple_of(59) {
                (600 + (idx % 200)) as i16
            } else {
                (idx % 64) as i16
            }
        }))
        .into_array();
        let large_packed = BitPackedData::encode(&large_values, 6, &mut ctx)?;
        let mut rng = StdRng::seed_from_u64(123);
        let random_raw: Vec<u32> = (0..large_len / 32)
            .map(|_| rng.random_range(0..large_len as u32))
            .collect();
        let mut sorted_raw = random_raw.clone();
        sorted_raw.sort_unstable();

        for index_ptype in [
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I16,
            PType::I32,
            PType::I64,
        ] {
            for (raw, expect_full_decode) in [(&sorted_raw, false), (&random_raw, true)] {
                let indices_prim = PrimitiveArray::from_iter(raw.iter().copied())
                    .into_array()
                    .cast(index_ptype.into())?
                    .execute::<PrimitiveArray>(&mut ctx)?;
                let patches = large_packed.patches().expect("large array has patches");
                assert_eq!(
                    should_decode_full_for_patches(
                        large_packed.as_view(),
                        &indices_prim,
                        &patches,
                    )?,
                    expect_full_decode
                );
                let indices = indices_prim.into_array();
                assert_arrays_eq!(
                    large_packed.clone().into_array().take(indices.clone())?,
                    large_values.take(indices)?,
                    &mut ctx
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_take_patched_constant_values_and_narrow_chunk_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 4 * FL_CHUNK_SIZE;
        let values = PrimitiveArray::from_iter((0..len).map(|idx| {
            if idx.is_multiple_of(31) {
                999i16
            } else {
                (idx % 64) as i16
            }
        }))
        .into_array();
        let encoded = BitPackedData::encode(&values, 6, &mut ctx)?;
        let parts = BitPacked::into_parts(encoded);
        let orig_patches = parts.patches.expect("encoded array has patches");
        let const_patches = Patches::new(
            orig_patches.array_len(),
            orig_patches.offset(),
            orig_patches.indices().clone(),
            ConstantArray::new(999i16, orig_patches.num_patches()).into_array(),
            orig_patches
                .chunk_offsets()
                .as_ref()
                .map(|co| co.cast(PType::U16.into()))
                .transpose()?,
            &mut ctx,
        )?;
        let packed = BitPacked::try_new(
            parts.packed,
            PType::I16,
            parts.validity,
            Some(const_patches),
            6,
            parts.len,
            parts.offset,
        )?;
        let indices =
            PrimitiveArray::from_iter([0u32, 1, 31, 62, 1024, 1054, 2048, 3069, 4092]).into_array();
        assert_arrays_eq!(
            packed.into_array().take(indices.clone())?,
            values.take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_patched_piecewise_unsorted_ranges() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 32_768usize;
        let values = PrimitiveArray::from_iter((0..len).map(|idx| {
            if idx.is_multiple_of(43) {
                (1_000 + (idx % 500)) as i16
            } else {
                (idx % 64) as i16
            }
        }))
        .into_array();
        let packed = BitPackedData::encode(&values, 6, &mut ctx)?.into_array();

        // Unsorted overlapping runs totalling 3,072 rows (< 32,768 / 8 = 4,096).
        let indices = PiecewiseSequenceArray::try_new(
            buffer![20_000u64, 5_000, 12_000, 5_100, 28_000, 1_000].into_array(),
            ConstantArray::new(512u64, 6).into_array(),
            ConstantArray::new(1u64, 6).into_array(),
            3_072,
        )?
        .into_array();
        let expanded = indices
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(packed.take(indices)?, values.take(expanded)?, &mut ctx);
        Ok(())
    }
}
