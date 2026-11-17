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
use fastlanes::Delta as FastLanesDelta;
use num_traits::AsPrimitive;
use num_traits::WrappingAdd;
use num_traits::WrappingSub;
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
    /// Where the probe's lane reads find the deltas and their patches, resolved on its first lane
    /// read; `Some(None)` when lane reads do not apply to the array.
    lanes: Option<Option<ResolvedLanes>>,
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
                && let Some(lanes) = retained
                    .lanes
                    .get_or_insert_with(|| ResolvedLanes::resolve(array))
                && let Some(value) = lane_value(array, Some(lanes), chunk, within, ctx)?
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
        None => match lane_value(array, None, chunk, within, ctx)? {
            Some(value) => Ok(value),
            None => Ok(value_at(array, &decode_chunk(array, chunk, ctx)?, within)),
        },
    }
}

/// The value at `within` of chunk `chunk`, summed from its lane's base and the deltas of its lane
/// The value at `within` of chunk `chunk`, read from its lane alone, or `None` when lane reads do
/// not apply to the array's deltas. A repeated probe passes the deltas it resolved before in
/// `resolved`; a one-off read finds them in `array`.
fn lane_value(
    array: ArrayView<'_, Delta>,
    resolved: Option<&ResolvedLanes>,
    chunk: usize,
    within: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let ptype = array.dtype().as_ptype();
    let nullability = array.dtype().nullability();
    let Some(bits) = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
        lane_value_bits::<T>(array, resolved, chunk, within, ctx)?.map(|value| value.as_())
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
    resolved: Option<&ResolvedLanes>,
    chunk: usize,
    within: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<T>>
where
    T: NativePType + FastLanesDelta + WrappingAdd + WrappingSub + AsPrimitive<u64>,
{
    let (deltas, patches) = match resolved {
        Some(resolved) => {
            let Some(deltas) = resolved.deltas::<T>(chunk) else {
                return Ok(None);
            };
            let patches = resolved.patches.as_ref().map(ResolvedPatches::as_ref);
            (deltas, PatchSource::Resolved(patches))
        }
        None => {
            let deltas = array.deltas();
            if let Some(plain) = deltas.as_opt::<Primitive>() {
                let Some(deltas) = LaneDeltas::plain(plain, chunk) else {
                    return Ok(None);
                };
                (deltas, PatchSource::Resolved(None))
            } else {
                let Some(packed) = PackedDeltas::try_new(deltas) else {
                    return Ok(None);
                };
                let bp = packed.packed();
                let Some(deltas) = LaneDeltas::packed(bp, packed.first_block() + chunk) else {
                    return Ok(None);
                };
                (deltas, PatchSource::Find(bp))
            }
        }
    };
    // Delta works on the transposed chunk, so find where the value sits in it.
    let position = fastlanes::untranspose(within);
    let (lane, _) = fastlanes::delta_lane_row::<T>(position);
    let base = base_bits::<T>(array.bases(), chunk * T::LANES + lane, ctx)?;
    Ok(deltas.undelta(base, position, patches))
}

/// Where a repeated probe's lane reads find a Delta array's deltas and their patches, resolved
/// once on its first lane read.
///
/// A one-off read looks these up in the array itself, which takes slot lookups, unwrapping a
/// slice of the deltas, and rebuilding the patches. Holding clones of the arrays costs atomic
/// reference counting, which a one-off read would not earn back, so only repeated probes keep
/// them.
pub(crate) struct ResolvedLanes {
    deltas: ResolvedDeltas,
    patches: Option<ResolvedPatches>,
}

/// The deltas a repeated probe reads.
enum ResolvedDeltas {
    /// Plain deltas on the host.
    Plain(ArrayRef),
    /// Deltas bit-packed on the host, starting at block `first_block` of the `BitPacked` array
    /// `packed`.
    Packed {
        packed: ArrayRef,
        first_block: usize,
    },
}

/// The patches of a repeated probe's bit-packed deltas.
struct ResolvedPatches {
    indices: ArrayRef,
    values: ArrayRef,
    offset: usize,
    shift: usize,
}

impl ResolvedPatches {
    fn as_ref(&self) -> PatchesRef<'_> {
        PatchesRef {
            indices: &self.indices,
            values: &self.values,
            offset: self.offset,
            shift: self.shift,
        }
    }
}

impl ResolvedLanes {
    /// Where lane reads find the deltas of `array`, or `None` when they are not plain or
    /// bit-packed on the host.
    fn resolve(array: ArrayView<'_, Delta>) -> Option<Self> {
        let deltas = array.deltas();
        if let Some(plain) = deltas.as_opt::<Primitive>() {
            return plain.buffer_handle().is_on_host().then(|| Self {
                deltas: ResolvedDeltas::Plain(deltas.clone()),
                patches: None,
            });
        }
        let packed = PackedDeltas::try_new(deltas)?;
        let bp = packed.packed();
        let patches = bp.patches().map(|patches| ResolvedPatches {
            indices: patches.indices().clone(),
            values: patches.values().clone(),
            offset: patches.offset(),
            shift: usize::from(bp.offset()),
        });
        Some(Self {
            deltas: ResolvedDeltas::Packed {
                packed: bp.array().clone(),
                first_block: packed.first_block(),
            },
            patches,
        })
    }

    /// The deltas of chunk `chunk`.
    fn deltas<T: NativePType + BitPacking>(&self, chunk: usize) -> Option<LaneDeltas<'_, T>> {
        match &self.deltas {
            ResolvedDeltas::Plain(deltas) => {
                LaneDeltas::plain(deltas.as_opt::<Primitive>()?, chunk)
            }
            ResolvedDeltas::Packed {
                packed,
                first_block,
            } => LaneDeltas::packed(packed.as_opt::<BitPacked>()?, first_block + chunk),
        }
    }
}

/// The patches of bit-packed deltas, borrowed for one read.
#[derive(Clone, Copy)]
struct PatchesRef<'a> {
    indices: &'a ArrayRef,
    values: &'a ArrayRef,
    /// The patches' offset, which their indices count from.
    offset: usize,
    /// How far into its first block the bit-packed array starts.
    shift: usize,
}

/// Where a lane read finds the patches of its bit-packed deltas.
#[derive(Clone, Copy)]
enum PatchSource<'a> {
    /// A one-off read looks them up in the bit-packed array, and only when it reaches them.
    Find(ArrayView<'a, BitPacked>),
    /// A repeated probe resolved them before, or there are none.
    Resolved(Option<PatchesRef<'a>>),
}

/// The deltas of one chunk.
enum LaneDeltas<'a, T> {
    /// Plain deltas of the chunk.
    Plain(&'a [T; FL_CHUNK_SIZE]),
    /// The packed block of the chunk, its bit width, and the block's index in the bit-packed
    /// array, which its patches count by.
    Packed {
        block: &'a [T],
        width: usize,
        block_index: usize,
    },
}

impl<'a, T: NativePType + BitPacking> LaneDeltas<'a, T> {
    /// The deltas of chunk `chunk` of plain deltas, when they are on the host.
    // Inlined into both the one-off and the cached read, so the one-off read costs what it did
    // before the cache: out of line it is about 6% more instructions per read.
    #[expect(
        clippy::inline_always,
        reason = "keeps the one-off read as cheap as without the cache"
    )]
    #[inline(always)]
    fn plain(deltas: ArrayView<'a, Primitive>, chunk: usize) -> Option<Self> {
        let values = host_values::<T>(deltas)?;
        let values = values.get(chunk * FL_CHUNK_SIZE..(chunk + 1) * FL_CHUNK_SIZE)?;
        Some(Self::Plain(values.try_into().ok()?))
    }

    /// Block `block` of bit-packed deltas, when it is on the host.
    // Inlined into both the one-off and the cached read, so the one-off read costs what it did
    // before the cache: out of line it is about 6% more instructions per read.
    #[expect(
        clippy::inline_always,
        reason = "keeps the one-off read as cheap as without the cache"
    )]
    #[inline(always)]
    fn packed(bp: ArrayView<'a, BitPacked>, block: usize) -> Option<Self> {
        if !bp.packed().is_on_host() {
            return None;
        }
        let words_of = |width: usize| FL_CHUNK_SIZE * width / (8 * size_of::<T>());
        let all = bp.data().packed_slice::<T>();
        match bp.bit_widths() {
            BitWidthsView::Global(width) => {
                let width = usize::from(width);
                let words = words_of(width);
                Some(Self::Packed {
                    block: all.get(block * words..(block + 1) * words)?,
                    width,
                    block_index: block,
                })
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
                Some(Self::Packed {
                    block: packed_block,
                    width,
                    block_index: block,
                })
            }
        }
    }

    /// The value at `position` of the undeltaed chunk, whose lane has base `base`, corrected for
    /// the patches from `patches`, or `None` when the patches are not in plain host arrays.
    fn undelta(&self, base: T, position: usize, patches: PatchSource<'_>) -> Option<T>
    where
        T: FastLanesDelta + WrappingAdd + WrappingSub,
    {
        match self {
            Self::Plain(values) => Some(T::undelta_single(values, base, position)),
            &Self::Packed {
                block,
                width,
                block_index,
            } => {
                let value = T::undelta_pack_single(width, block, base, position);
                let correction = match patches {
                    PatchSource::Resolved(None) => return Some(value),
                    PatchSource::Resolved(Some(patches)) => {
                        patch_correction(patches, block_index, block, width, position)?
                    }
                    PatchSource::Find(bp) => {
                        let Some(patches) = bp.patches() else {
                            return Some(value);
                        };
                        let patches = PatchesRef {
                            indices: patches.indices(),
                            values: patches.values(),
                            offset: patches.offset(),
                            shift: usize::from(bp.offset()),
                        };
                        patch_correction(patches, block_index, block, width, position)?
                    }
                };
                Some(value.wrapping_add(&correction))
            }
        }
    }
}

/// What `patches` add to the undeltaed value at `position` of packed block `block_index`, whose
/// words `block` hold the deltas at `width` bits: for each patch in the value's lane up to its
/// row, the patch value less the packed delta it replaces. `None` when the patches are not in plain
/// host arrays.
// Inlined into both the one-off and the cached read, like `LaneDeltas::packed`.
#[expect(
    clippy::inline_always,
    reason = "keeps the one-off read as cheap as without the cache"
)]
#[inline(always)]
fn patch_correction<T>(
    patches: PatchesRef<'_>,
    block_index: usize,
    block: &[T],
    width: usize,
    position: usize,
) -> Option<T>
where
    T: NativePType + BitPacking + WrappingAdd + WrappingSub,
{
    let values = host_values::<T>(patches.values.as_opt::<Primitive>()?)?;
    let indices = patches.indices.as_opt::<Primitive>()?;
    if !indices.buffer_handle().is_on_host() {
        return None;
    }
    // Patch indices count from the patches' offset and from the start of the bit-packed array,
    // which sits `shift` into its first block; positions here count from the start of the block.
    let shift = patches.shift;
    let block_start = block_index * FL_CHUNK_SIZE + patches.offset;
    let (lane, row) = fastlanes::delta_lane_row::<T>(position);
    let mut correction = T::zero();
    let mut correct = |patched: usize, delta: T| {
        // SAFETY: `LaneDeltas::packed` checked the block holds 1024 values of `width` bits, and
        // `patched` is within the block.
        let packed = unsafe { BitPacking::unchecked_unpack_single(width, block, patched) };
        correction = correction.wrapping_add(&delta).wrapping_sub(&packed);
    };
    match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
        let indices = indices.as_slice::<P>();
        let index_of = |index: P| AsPrimitive::<usize>::as_(index) + shift;
        let lo = indices.partition_point(|&index| index_of(index) < block_start);
        let hi = indices.partition_point(|&index| index_of(index) < block_start + FL_CHUNK_SIZE);
        let (indices, values) = (&indices[lo..hi], &values[lo..hi]);
        if indices.len() <= row + 1 {
            // Few patches: check each one's lane and row.
            for (&index, &delta) in indices.iter().zip(values) {
                let patched = index_of(index) - block_start;
                let (patch_lane, patch_row) = fastlanes::delta_lane_row::<T>(patched);
                if patch_lane == lane && patch_row <= row {
                    correct(patched, delta);
                }
            }
        } else {
            // Many patches: look up each position of the lane prefix.
            for k in 0..=row {
                let patched = fastlanes::delta_index::<T>(lane, k);
                let target = block_start + patched;
                if let Ok(i) = indices.binary_search_by(|&index| index_of(index).cmp(&target)) {
                    correct(patched, values[i]);
                }
            }
        }
    });
    Some(correction)
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
