// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use super::gather::U8_LANES;
use super::gather::gather_chunk;
use super::gather::gather_delta_lanes;
use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::Delta;
use crate::DeltaArrayExt;
use crate::DeltaArraySlotsExt;
use crate::FL_CHUNK_SIZE;
use crate::RLEArray;
use crate::bitpack_decompress::execute_block_offsets;
use crate::bitpack_decompress::for_each_packed_block;
use crate::rle::RLEArrayExt;
use crate::rle::RLEArraySlotsExt;
use crate::unpack_iter::for_each_packed_chunk;

/// Decompresses an RLE array back into a primitive array.
pub fn rle_decompress(array: &RLEArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    // The per-chunk value-index offsets are tiny (one entry per 1024-element chunk), so cast them
    // to `u64` once here instead of monomorphizing the whole decode loop over the offset width.
    let values_idx_offsets = array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let values_idx_offsets: Vec<u64> =
        match_each_unsigned_integer_ptype!(values_idx_offsets.ptype(), |O| {
            values_idx_offsets
                .as_slice::<O>()
                .iter()
                .map(|&o| o.as_())
                .collect()
        });

    match_each_native_ptype!(array.values().dtype().as_ptype(), |V| {
        // RLE indices are always u16 (or u8 if downcasted).
        match array.indices().dtype().as_ptype() {
            PType::U8 => rle_decode_typed::<V, u8>(array, &values_idx_offsets, ctx),
            PType::U16 => rle_decode_typed::<V, u16>(array, &values_idx_offsets, ctx),
            _ => vortex_panic!(
                "Unsupported index type for RLE decoding: {}",
                array.indices().dtype().as_ptype()
            ),
        }
    })
}

/// Decompresses an `RLEArray` into to a primitive array of unsigned integers.
///
/// The indices are decoded in one of three ways, from the cheapest:
///
/// - Indices stored as `u8` FastLanes Delta over 0- or 1-bit deltas, which is how the compressor
///   stores RLE indices when Delta is enabled, are expanded straight from the delta bits by
///   [`gather_delta_lanes`], without materializing the indices.
/// - BitPacked indices are unpacked one chunk at a time into a stack buffer and gathered while
///   that chunk is in L1, instead of unpacking every index first.
/// - Any other indices are executed to a primitive array and gathered.
fn rle_decode_typed<V, I>(
    array: &RLEArray,
    values_idx_offsets: &[u64],
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    V: NativePType,
    I: NativePType + PhysicalPType<Physical = I> + BitPacking + Ord + Into<usize>,
{
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<V>();

    // The offsets come from (possibly untrusted) storage. Validate them once here so
    // that the per-chunk slicing and the unchecked decodes below stay within `values`:
    // offsets must be non-decreasing and span at most `values.len()` values.
    vortex_ensure!(
        values_idx_offsets.is_sorted(),
        "RLE values_idx_offsets must be non-decreasing"
    );
    if let (Some(&first), Some(&last)) = (values_idx_offsets.first(), values_idx_offsets.last()) {
        vortex_ensure!(
            last - first <= values.len() as u64,
            "RLE values_idx_offsets span {} values but only {} are present",
            last - first,
            values.len()
        );
    }

    let chunk_start_idx = array.offset() / FL_CHUNK_SIZE;
    let chunk_end_idx = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);
    let num_chunks = chunk_end_idx - chunk_start_idx;
    let chunks = ChunkValues {
        values,
        values_idx_offsets,
        num_chunks,
    };

    let mut buffer = BufferMut::<V>::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let (out_buf, _) = buffer.spare_capacity_mut().as_chunks_mut::<FL_CHUNK_SIZE>();
    // SAFETY: `MaybeUninit<T>` and `T` have the same layout, and every chunk below is written
    // in full before the buffer is read.
    let out_buf: &mut [[V; FL_CHUNK_SIZE]] = unsafe { std::mem::transmute(out_buf) };

    let indices = array.indices().clone();
    assert!(indices.len().is_multiple_of(FL_CHUNK_SIZE));
    let decoded = size_of::<I>() == 1
        && match DeltaIndices::try_new(&indices, ctx)? {
            Some(delta) => delta.decode(&chunks, out_buf)?,
            None => false,
        };
    if !decoded {
        match ChunkedBitPacked::try_new(indices) {
            Ok(bp) => decode_bitpacked_indices::<V, I>(&bp, &chunks, out_buf, ctx)?,
            Err(indices) => decode_primitive_indices::<V, I>(indices, &chunks, out_buf, ctx)?,
        }
    }

    unsafe {
        buffer.set_len(num_chunks * FL_CHUNK_SIZE);
    }

    let offset_within_chunk = array.offset();

    Ok(PrimitiveArray::new(
        buffer
            .freeze()
            .slice(offset_within_chunk..(offset_within_chunk + array.len())),
        array.validity()?,
    ))
}

/// The run values and per-chunk value offsets of the chunks being decoded.
struct ChunkValues<'a, V> {
    values: &'a [V],
    values_idx_offsets: &'a [u64],
    num_chunks: usize,
}

impl<'a, V> ChunkValues<'a, V> {
    /// The values from the start of chunk `chunk_idx` to the end of all values, and how many of
    /// them belong to the chunk.
    fn chunk(&self, chunk_idx: usize) -> VortexResult<(&'a [V], u16)> {
        // Offsets in `values_idx_offsets` are absolute and need to be shifted
        // by the offset of the first chunk, respective of the current slice,
        // to make them relative.
        let base = self.values_idx_offsets[0];
        let value_idx_offset = (self.values_idx_offsets[chunk_idx] - base) as usize;
        let next_value_idx_offset = if chunk_idx + 1 < self.num_chunks {
            (self.values_idx_offsets[chunk_idx + 1] - base) as usize
        } else {
            self.values.len()
        };
        let num_chunk_values = u16::try_from(next_value_idx_offset - value_idx_offset)
            .vortex_expect("There can be at most 1024 values in RLE chunk");
        vortex_ensure!(
            num_chunk_values > 0,
            "RLE chunk {chunk_idx} references no values"
        );
        Ok((&self.values[value_idx_offset..], num_chunk_values))
    }
}

/// Decode one chunk from its indices in original order.
///
/// `validity_bits` is the validity of all indices, or `None` when every index is valid.
fn decode_chunk<V, I>(
    chunks: &ChunkValues<'_, V>,
    chunk_idx: usize,
    chunk_indices: &[I; FL_CHUNK_SIZE],
    validity_bits: Option<&BitBuffer>,
    out: &mut [V; FL_CHUNK_SIZE],
) -> VortexResult<()>
where
    V: Copy,
    I: NativePType + Ord + Into<usize>,
{
    let (chunk_values, num_chunk_values) = chunks.chunk(chunk_idx)?;
    if num_chunk_values == 1 {
        // Single-value chunk: fill directly to avoid out-of-bounds index
        // access. The indices may contain values other than 0 when they
        // have been further compressed (e.g., as a masked constant).
        out.fill(chunk_values[0]);
        return Ok(());
    }
    let chunk_start = chunk_idx * FL_CHUNK_SIZE;
    let chunk_valid_count = validity_bits.map_or(FL_CHUNK_SIZE, |bits| {
        bits.count_range(chunk_start, chunk_start + FL_CHUNK_SIZE)
    });
    if chunk_valid_count == FL_CHUNK_SIZE {
        decode_chunk_checked(
            chunk_values,
            chunk_indices,
            num_chunk_values,
            chunk_idx,
            out,
        )
    } else if chunk_valid_count == 0 {
        // Entirely-null chunk: every position is masked out, so the indices are
        // meaningless (possibly garbage after further compression) — skip the
        // gather and emit an arbitrary in-bounds value.
        out.fill(chunk_values[0]);
        Ok(())
    } else {
        // Null positions may contain arbitrary garbage indices after further
        // compression, so zero them out (their decoded values are masked by the
        // validity). Valid positions must be genuinely in bounds and are still
        // bound-checked before the unchecked gather: a corrupt index at a valid
        // position is an error, never silently remapped.
        let bits = validity_bits.vortex_expect("mixed-validity chunk implies a materialized mask");
        let mut sanitized: [u16; FL_CHUNK_SIZE] = [0; FL_CHUNK_SIZE];
        bits.slice(chunk_start..chunk_start + FL_CHUNK_SIZE)
            .for_each_set_index(|i| {
                sanitized[i] = num_traits::cast::<I, u16>(chunk_indices[i])
                    .vortex_expect("RLE indices are always less than u16");
            });
        decode_chunk_checked(chunk_values, &sanitized, num_chunk_values, chunk_idx, out)
    }
}

/// Decode every chunk from indices executed to a primitive array.
fn decode_primitive_indices<V, I>(
    indices: ArrayRef,
    chunks: &ChunkValues<'_, V>,
    out_buf: &mut [[V; FL_CHUNK_SIZE]],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    V: NativePType,
    I: NativePType + Ord + Into<usize>,
{
    let indices = indices.execute::<PrimitiveArray>(ctx)?;
    let validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    // `None` means every position is valid.
    let validity_bits = (!validity.all_true()).then(|| validity.to_bit_buffer());
    let (indices_sl, _) = indices.as_slice::<I>().as_chunks::<FL_CHUNK_SIZE>();
    for (chunk_idx, (chunk_indices, out)) in indices_sl.iter().zip(out_buf.iter_mut()).enumerate() {
        decode_chunk(
            chunks,
            chunk_idx,
            chunk_indices,
            validity_bits.as_ref(),
            out,
        )?;
    }
    Ok(())
}

/// Whole chunks of a BitPacked array without patches: either the array itself, or a lazy slice of
/// it that starts and ends on chunk boundaries.
struct ChunkedBitPacked {
    array: BitPackedArray,
    /// The chunks of `array` in use.
    chunks: Range<usize>,
}

impl ChunkedBitPacked {
    /// Recognize BitPacked indices, possibly sliced, that decode one whole chunk at a time, or
    /// give the array back.
    fn try_new(array: ArrayRef) -> Result<Self, ArrayRef> {
        let (array, range) = match array.try_downcast::<Slice>() {
            Ok(slice) => {
                let range = slice.data().slice_range().clone();
                match slice.child().clone().try_downcast::<BitPacked>() {
                    // A global-width slice reduces to a BitPacked array with its own offset.
                    Ok(bp) => {
                        match <BitPacked as SliceReduce>::slice(bp.as_view(), range.clone()) {
                            Ok(Some(reduced)) => {
                                let reduced = reduced.try_downcast::<BitPacked>()?;
                                let len = reduced.len();
                                (reduced, 0..len)
                            }
                            _ => (bp, range),
                        }
                    }
                    Err(child) => return Err(SliceArray::new(child, range).into_array()),
                }
            }
            Err(array) => {
                let bp = array.try_downcast::<BitPacked>()?;
                let len = bp.len();
                (bp, 0..len)
            }
        };
        let fits = array.offset() == 0
            && array.patches().is_none()
            && range.start.is_multiple_of(FL_CHUNK_SIZE)
            && range.len().is_multiple_of(FL_CHUNK_SIZE);
        if !fits {
            let array = array.into_array();
            return Err(if range == (0..array.len()) {
                array
            } else {
                SliceArray::new(array, range).into_array()
            });
        }
        let chunks = range.start / FL_CHUNK_SIZE..range.end / FL_CHUNK_SIZE;
        Ok(Self { array, chunks })
    }

    /// Whether this covers the whole array.
    fn is_whole(&self) -> bool {
        self.chunks.start == 0 && self.chunks.end * FL_CHUNK_SIZE == self.array.len()
    }
}

/// Decode every chunk from BitPacked indices, unpacking one chunk at a time.
fn decode_bitpacked_indices<V, I>(
    bp: &ChunkedBitPacked,
    chunks: &ChunkValues<'_, V>,
    out_buf: &mut [[V; FL_CHUNK_SIZE]],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    V: NativePType,
    I: NativePType + PhysicalPType<Physical = I> + BitPacking + Ord + Into<usize>,
{
    let array = &bp.array;
    let len = array.len();
    let used = bp.chunks.start * FL_CHUNK_SIZE..bp.chunks.end * FL_CHUNK_SIZE;
    let validity = array
        .validity()?
        .slice(used.clone())?
        .execute_mask(used.len(), ctx)?;
    let validity_bits = (!validity.all_true()).then(|| validity.to_bit_buffer());
    let packed = array.packed_slice::<I>();

    let mut scratch = [I::default(); FL_CHUNK_SIZE];
    let mut result = Ok(());
    let mut decode_packed = |packed: &[I], bit_width: usize, range: Range<usize>| {
        let Some(chunk_idx) = (range.start / FL_CHUNK_SIZE).checked_sub(bp.chunks.start) else {
            return;
        };
        if result.is_err() || chunk_idx >= out_buf.len().min(bp.chunks.len()) {
            return;
        }
        // SAFETY: the BitPacked helpers hand out one whole chunk packed at `bit_width`, and
        // `scratch` holds a whole chunk.
        unsafe { I::unchecked_unpack(bit_width, packed, &mut scratch) };
        result = decode_chunk(
            chunks,
            chunk_idx,
            &scratch,
            validity_bits.as_ref(),
            &mut out_buf[chunk_idx],
        );
    };
    match array.bit_widths() {
        BitWidthsView::Global(bit_width) => {
            let bit_width = usize::from(bit_width);
            for_each_packed_chunk::<I, _>(packed, bit_width, 0, len, |packed, range| {
                decode_packed(packed, bit_width, range)
            })?;
        }
        BitWidthsView::Blocked(offsets) => {
            let offsets = execute_block_offsets(array.as_view(), offsets, ctx)?;
            for_each_packed_block::<I, _>(packed, &offsets, 0, len, decode_packed);
        }
    }
    result
}

/// `u8` RLE indices stored as FastLanes Delta whose deltas are all 0 or 1.
struct DeltaIndices {
    bases: PrimitiveArray,
    /// One byte of delta bits per lane, or `None` when every delta is zero.
    deltas: Option<BitPackedArray>,
    /// Whether every index is valid.
    all_valid: bool,
}

impl DeltaIndices {
    /// Recognize Delta indices this decoder can expand directly, or return `None`.
    fn try_new(indices: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        let Ok(delta) = indices.clone().try_downcast::<Delta>() else {
            return Ok(None);
        };
        if delta.offset() != 0 || delta.dtype().as_ptype() != PType::U8 {
            return Ok(None);
        }
        let all_valid = matches!(
            delta.validity()?,
            Validity::NonNullable | Validity::AllValid
        );
        let padded_len = delta.len().next_multiple_of(FL_CHUNK_SIZE);
        let deltas = match ChunkedBitPacked::try_new(delta.deltas().clone()) {
            Ok(bp) => {
                let fits = bp.is_whole()
                    && bp.array.len() == padded_len
                    && matches!(bp.array.bit_widths(), BitWidthsView::Global(0 | 1));
                if !fits {
                    return Ok(None);
                }
                matches!(bp.array.bit_widths(), BitWidthsView::Global(1)).then_some(bp.array)
            }
            Err(deltas) => match deltas.try_downcast::<Constant>() {
                Ok(constant) if constant.scalar().as_primitive().typed_value::<u8>() == Some(0) => {
                    None
                }
                _ => return Ok(None),
            },
        };
        let bases = delta.bases().clone().execute::<PrimitiveArray>(ctx)?;
        if bases.ptype() != PType::U8 || bases.len() != padded_len / FL_CHUNK_SIZE * U8_LANES {
            return Ok(None);
        }
        Ok(Some(Self {
            bases,
            deltas,
            all_valid,
        }))
    }

    /// Decode every chunk, or return `false` to have the caller decode the indices another way.
    ///
    /// Null positions repeat the preceding valid index, so they are decoded like the rest and
    /// masked by the validity. If corrupt data puts an out-of-bounds index under a null, the
    /// indices are left to the caller, which ignores null positions.
    fn decode<V: Copy>(
        &self,
        chunks: &ChunkValues<'_, V>,
        out_buf: &mut [[V; FL_CHUNK_SIZE]],
    ) -> VortexResult<bool> {
        let (bases, _) = self.bases.as_slice::<u8>().as_chunks::<U8_LANES>();
        let deltas = self
            .deltas
            .as_ref()
            .map(|bp| bp.packed_slice::<u8>().as_chunks::<U8_LANES>().0);
        for (chunk_idx, out) in out_buf.iter_mut().enumerate() {
            let (chunk_values, num_chunk_values) = chunks.chunk(chunk_idx)?;
            if num_chunk_values == 1 {
                out.fill(chunk_values[0]);
                continue;
            }
            let chunk_bases = &bases[chunk_idx];
            let chunk_deltas = deltas.map(|d| &d[chunk_idx]);
            let max_index = max_lane_index(chunk_bases, chunk_deltas);
            let in_bounds =
                max_index < usize::from(num_chunk_values) && max_index <= usize::from(u8::MAX);
            if !in_bounds {
                if !self.all_valid {
                    return Ok(false);
                }
                vortex_bail!(
                    "RLE index {max_index} out of bounds for chunk {chunk_idx} with {num_chunk_values} values"
                );
            }
            // SAFETY: every lane's last index is below `num_chunk_values`, which the offset
            // validation bounds by `chunk_values.len()`.
            unsafe { gather_delta_lanes(chunk_values, chunk_bases, chunk_deltas, out) };
        }
        Ok(true)
    }
}

/// The largest last index of any lane: indices only grow along a lane, so a lane's last index is
/// its base plus its number of set delta bits.
///
/// The common case reduces in `u8` with saturating adds, which vectorizes; only a saturated result
/// is recomputed exactly, so indices of 255 and beyond are told apart.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a byte has at most 8 set bits"
)]
fn max_lane_index(bases: &[u8; U8_LANES], deltas: Option<&[u8; U8_LANES]>) -> usize {
    let saturating = match deltas {
        None => bases.iter().copied().max(),
        Some(deltas) => bases
            .iter()
            .zip(deltas)
            .map(|(&base, &bits)| base.saturating_add(bits.count_ones() as u8))
            .max(),
    }
    .unwrap_or_default();
    if saturating < u8::MAX {
        return usize::from(saturating);
    }
    bases
        .iter()
        .enumerate()
        .map(|(lane, &base)| {
            usize::from(base) + deltas.map_or(0, |d| d[lane].count_ones() as usize)
        })
        .max()
        .unwrap_or_default()
}

/// Bound-checks every index in the chunk, then runs the unchecked fastlanes gather.
///
/// The indices come from (possibly untrusted) storage: a single max-reduction over the
/// chunk vectorizes and keeps the bounds check out of the per-element decode loop. The
/// reduction runs in the index type and widens only its result: widening every index to
/// `usize` first quarters the lanes per vector and made the check cost as much as the gather.
fn decode_chunk_checked<V, I>(
    chunk_values: &[V],
    chunk_indices: &[I; FL_CHUNK_SIZE],
    num_chunk_values: u16,
    chunk_idx: usize,
    out: &mut [V; FL_CHUNK_SIZE],
) -> VortexResult<()>
where
    V: Copy,
    I: Copy + Ord + Into<usize>,
{
    let max_index: usize = chunk_indices.iter().copied().max().map_or(0, Into::into);
    vortex_ensure!(
        max_index < num_chunk_values as usize,
        "RLE index {max_index} out of bounds for chunk {chunk_idx} with {num_chunk_values} values"
    );
    // SAFETY: just checked that every index in the chunk is below `num_chunk_values`,
    // which the caller's offset validation bounds by `chunk_values.len()`.
    unsafe { gather_chunk(chunk_values, chunk_indices, out) };
    Ok(())
}

#[cfg(test)]
mod tests;
