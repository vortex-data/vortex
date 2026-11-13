// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Delta scalar access.
//!
//! Delta transposes each 1024-value chunk into lanes and encodes each lane against its own base,
//! so a value depends only on its lane's base and the deltas of its lane up to its row: at most 8
//! to 64 deltas, by integer width. A read of plain or bit-packed deltas, patched or not, sums just
//! those. A repeated probe that returns to a chunk decodes the whole chunk and keeps it, so reads
//! that stay within a chunk decode it once. Deltas in any other encoding, such as frame of
//! reference, decode the chunk instead.

use fastlanes::BitPacking;
use fastlanes::FL_ORDER;
use fastlanes::FastLanes;
use num_traits::AsPrimitive;
use num_traits::WrappingAdd;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ProbeState;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::delta_decompress::PackedDeltas;
use crate::delta::array::lane_count;

/// What a repeated probe keeps between reads.
#[derive(Default)]
pub struct DeltaProbeState {
    /// The last chunk the probe decoded, and its values in logical order.
    chunk: Option<(usize, PrimitiveArray)>,
    /// The last chunk the probe read a single lane of. A second read in it decodes the chunk.
    lane_read: Option<usize>,
}

pub(crate) fn scalar_at(
    state: &mut ProbeState<'_, Delta>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array();
    if !state.is_valid(index, ctx)? {
        return Ok(Scalar::null(array.dtype().clone()));
    }
    let position = array.offset() + index;
    let chunk = position / FL_CHUNK_SIZE;
    let within = position % FL_CHUNK_SIZE;
    match state.retained() {
        Some(retained) => {
            if let Some((cached, decoded)) = &retained.chunk
                && *cached == chunk
            {
                return Ok(value_at(array, decoded, within));
            }
            if retained.lane_read != Some(chunk)
                && let Some(value) = lane_value(array, chunk, within, ctx)?
            {
                retained.lane_read = Some(chunk);
                return Ok(value);
            }
            let decoded = &retained
                .chunk
                .insert((chunk, decode_chunk(array, chunk, ctx)?))
                .1;
            Ok(value_at(array, decoded, within))
        }
        None => match lane_value(array, chunk, within, ctx)? {
            Some(value) => Ok(value),
            None => Ok(value_at(array, &decode_chunk(array, chunk, ctx)?, within)),
        },
    }
}

/// The value at `within` of chunk `chunk`, summed from its lane's base and the deltas of its lane
/// up to its row, or `None` when the deltas are neither plain nor bit-packed on the host.
fn lane_value(
    array: ArrayView<'_, Delta>,
    chunk: usize,
    within: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let ptype = array.dtype().as_ptype();
    let nullability = array.dtype().nullability();
    let Some(bits) = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
        lane_value_bits::<T>(array, chunk, within, ctx)?.map(|value| value.as_())
    }) else {
        return Ok(None);
    };
    let bits: u64 = bits;
    // Truncating to the array's width reinterprets the unsigned sum as the array's type.
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    Ok(Some(match_each_integer_ptype!(ptype, |P| {
        Scalar::primitive::<P>(bits as P, nullability)
    })))
}

/// The bits of the value at `within` of chunk `chunk`, as the unsigned type `T` of the same width.
fn lane_value_bits<T>(
    array: ArrayView<'_, Delta>,
    chunk: usize,
    within: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<T>>
where
    T: NativePType + BitPacking + FastLanes + WrappingAdd + AsPrimitive<u64>,
{
    let lanes = T::LANES;
    // Delta lays a lane's row `k` at `FL_ORDER[k / 8] * 16 + (k % 8) * 128 + lane`; invert that
    // for the transposed position of the value.
    let position = untranspose(within);
    let in_row = position % 128;
    let lane = in_row % lanes;
    let row = FL_ORDER[(in_row - lane) / 16] * 8 + position / 128;

    let Some(deltas) = LaneDeltas::<T>::new(array.deltas(), chunk) else {
        return Ok(None);
    };
    let base = base_bits::<T>(array.bases(), chunk * lanes + lane, ctx)?;
    let mut value = base;
    for k in 0..=row {
        let index = FL_ORDER[k / 8] * 16 + (k % 8) * 128 + lane;
        value = value.wrapping_add(&deltas.get(index));
    }
    Ok(Some(value))
}

/// The position in a transposed chunk of the value at logical position `index`: the inverse of
/// [`fastlanes::transpose`].
fn untranspose(index: usize) -> usize {
    let lane = index / 64;
    // `FL_ORDER` is its own inverse.
    let order = FL_ORDER[(index / 8) % 8];
    let row = index % 8;
    row * 128 + order * 16 + lane
}

/// The deltas of one chunk, read one at a time.
enum LaneDeltas<'a, T> {
    /// Plain deltas of the chunk.
    Plain(&'a [T]),
    /// The packed block of the chunk, its bit width, and its patches as (position in chunk,
    /// value) in ascending position.
    Packed(&'a [T], usize, Vec<(usize, T)>),
}

impl<'a, T: NativePType + BitPacking> LaneDeltas<'a, T> {
    /// The deltas of chunk `chunk` when they are plain on the host, or bit-packed on the host
    /// from a block boundary with any patches in plain host arrays.
    fn new(deltas: &'a ArrayRef, chunk: usize) -> Option<Self> {
        if let Some(primitive) = deltas.as_opt::<Primitive>() {
            let values = host_values::<T>(primitive)?;
            let values = values.get(chunk * FL_CHUNK_SIZE..(chunk + 1) * FL_CHUNK_SIZE)?;
            return Some(Self::Plain(values));
        }
        let packed = PackedDeltas::try_new(deltas)?;
        let bp = packed.packed();
        if !bp.packed().is_on_host() {
            return None;
        }
        let words_of = |width: usize| FL_CHUNK_SIZE * width / (8 * size_of::<T>());
        let all = bp.data().packed_slice::<T>();
        let block = packed.first_block() + chunk;
        match bp.bit_widths() {
            BitWidthsView::Global(width) => {
                let width = usize::from(width);
                let words = words_of(width);
                let patches = block_patches(bp, block)?;
                Some(Self::Packed(
                    all.get(block * words..(block + 1) * words)?,
                    width,
                    patches,
                ))
            }
            BitWidthsView::Blocked(offsets) => {
                let offsets = offsets.as_opt::<Primitive>()?;
                if !offsets.buffer_handle().is_on_host() {
                    return None;
                }
                let (base, start, end): (u64, u64, u64) =
                    match_each_unsigned_integer_ptype!(offsets.ptype(), |I| {
                        let offsets = offsets.as_slice::<I>();
                        (
                            offsets.first()?.as_(),
                            offsets.get(block)?.as_(),
                            offsets.get(block + 1)?.as_(),
                        )
                    });
                let start = usize::try_from(start.checked_sub(base)?).ok()?;
                let end = usize::try_from(end.checked_sub(base)?).ok()?;
                let width = (end - start) / 128;
                let packed_block = all.get(start / size_of::<T>()..end / size_of::<T>())?;
                if packed_block.len() != words_of(width) {
                    return None;
                }
                Some(Self::Packed(packed_block, width, block_patches(bp, block)?))
            }
        }
    }

    /// The delta at `index` of the chunk.
    fn get(&self, index: usize) -> T {
        match self {
            Self::Plain(values) => values[index],
            Self::Packed(block, width, patches) => {
                match patches.binary_search_by_key(&index, |&(position, _)| position) {
                    Ok(patch) => patches[patch].1,
                    // SAFETY: `new` checked the block holds 1024 values of `width` bits, and
                    // `index` is within the chunk.
                    Err(_) => unsafe { BitPacking::unchecked_unpack_single(*width, block, index) },
                }
            }
        }
    }
}

/// The patches of `bp` in packed block `block`, as (position in block, value) in ascending
/// position, or `None` when they are not in plain host arrays.
fn block_patches<T: NativePType>(
    bp: ArrayView<'_, BitPacked>,
    block: usize,
) -> Option<Vec<(usize, T)>> {
    let Some(patches) = bp.patches() else {
        return Some(Vec::new());
    };
    let values = host_values::<T>(patches.values().as_opt::<Primitive>()?)?;
    let indices = patches.indices().as_opt::<Primitive>()?;
    if !indices.buffer_handle().is_on_host() {
        return None;
    }
    // Patch indices count from `patches.offset()` and from the start of `bp`, which sits
    // `bp.offset()` into its first block; positions here count from the start of that block.
    let shift = usize::from(bp.offset());
    let to_index = |position: usize| (position + patches.offset()).checked_sub(shift);
    let first = to_index(block * FL_CHUNK_SIZE).unwrap_or(0);
    let end = to_index((block + 1) * FL_CHUNK_SIZE)?;
    Some(match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
        let indices = indices.as_slice::<P>();
        let lo = indices.partition_point(|&index| AsPrimitive::<usize>::as_(index) < first);
        let hi = indices.partition_point(|&index| AsPrimitive::<usize>::as_(index) < end);
        indices[lo..hi]
            .iter()
            .zip(&values[lo..hi])
            .map(|(&index, &value)| {
                let position = AsPrimitive::<usize>::as_(index) + shift - patches.offset();
                (position - block * FL_CHUNK_SIZE, value)
            })
            .collect()
    }))
}

/// The values of a host-backed primitive array as `T`, the unsigned type of the same width.
fn host_values<T: NativePType>(primitive: ArrayView<'_, Primitive>) -> Option<&[T]> {
    if primitive.ptype().to_unsigned() != T::PTYPE {
        return None;
    }
    let bytes = primitive.buffer_handle().as_host_opt()?.as_slice();
    if !(bytes.as_ptr() as usize).is_multiple_of(align_of::<T>())
        || bytes.len() < primitive.len() * size_of::<T>()
    {
        return None;
    }
    // SAFETY: `T` has the width of the array's values, and the buffer is aligned for `T` and holds
    // `len` of them.
    Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), primitive.len()) })
}

/// The bits of base `index`, as the unsigned type `T` of the same width.
fn base_bits<T: NativePType>(
    bases: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<T> {
    if let Some(values) = bases.as_opt::<Primitive>().and_then(host_values::<T>) {
        return Ok(values[index]);
    }
    let scalar = bases.execute_scalar(index, ctx)?;
    let pvalue = scalar
        .as_primitive()
        .pvalue()
        .ok_or_else(|| vortex_err!("Delta base {index} is null"))?;
    pvalue.reinterpret_cast(T::PTYPE).cast::<T>()
}

/// Decode chunk `chunk` of `array` into its 1024 values, ignoring validity.
fn decode_chunk(
    array: ArrayView<'_, Delta>,
    chunk: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let lanes = lane_count(array.dtype().as_ptype());
    let bases = array.bases().slice(chunk * lanes..(chunk + 1) * lanes)?;
    let deltas = array
        .deltas()
        .slice(chunk * FL_CHUNK_SIZE..(chunk + 1) * FL_CHUNK_SIZE)?;
    Delta::try_new(bases, deltas, Validity::NonNullable, 0, FL_CHUNK_SIZE)?
        .into_array()
        .execute::<PrimitiveArray>(ctx)
}

/// The value at `within` of a decoded chunk, with the nullability of `array`.
fn value_at(array: ArrayView<'_, Delta>, decoded: &PrimitiveArray, within: usize) -> Scalar {
    let nullability = array.dtype().nullability();
    match_each_integer_ptype!(decoded.ptype(), |P| {
        Scalar::primitive::<P>(decoded.as_slice::<P>()[within], nullability)
    })
}

#[cfg(test)]
mod tests;
