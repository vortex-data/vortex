// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use fastlanes::Delta;
use fastlanes::FL_ORDER;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::slice::SliceSlots;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::DeltaArray;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_decompress::execute_block_offsets;
use crate::bitpack_decompress::for_each_packed_block;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::unpack_iter::for_each_packed_chunk;

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    let validity = array.validity()?;

    let original_ptype = bases.ptype();
    // Signed inputs are processed through their unsigned counterpart; `wrapping_add` on the
    // raw bytes inverts the `wrapping_sub` done at compress time regardless of signedness.
    let bases = bases.reinterpret_cast(original_ptype.to_unsigned());

    let packed = PackedDeltas::try_new(array.deltas());
    let deltas_len = array.deltas().len();

    let decoded = match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
        const LANES: usize = T::LANES;

        let buffer = match packed {
            Some(packed) => {
                fused_decompress_primitive::<T, LANES>(bases.as_slice(), packed, deltas_len, ctx)?
            }
            None => {
                let deltas = array
                    .deltas()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?
                    .reinterpret_cast(original_ptype.to_unsigned());
                decompress_primitive::<T, LANES>(bases.as_slice(), deltas.as_slice())
            }
        };
        // Whole chunks were decoded; keep the values within the Delta array's slice.
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    });

    Ok(decoded.reinterpret_cast(original_ptype))
}

/// Bit-packed deltas whose delta chunks each line up with one packed block.
///
/// Delta slices its deltas at chunk boundaries, so a BitPacked child, or a lazy slice of one, starts
/// on a packed block. The fused decode then reads whole packed blocks, and the Delta array's own
/// offset is applied by slicing the decoded chunks at the end.
#[derive(Clone, Copy)]
pub(crate) struct PackedDeltas<'a> {
    packed: ArrayView<'a, BitPacked>,
    /// The packed block holding the first delta.
    first_block: usize,
}

impl<'a> PackedDeltas<'a> {
    /// Returns the packed blocks behind `deltas` when it is a BitPacked array, or a slice of one,
    /// that starts on a packed block.
    pub(crate) fn try_new(deltas: &'a ArrayRef) -> Option<Self> {
        let (packed, slice_start) = match deltas.as_opt::<BitPacked>() {
            Some(packed) => (packed, 0),
            None => {
                let slice = deltas.as_opt::<Slice>()?;
                let start = slice.data().slice_range().start;
                let child = slice.slots()[SliceSlots::CHILD].as_ref()?;
                (child.as_opt::<BitPacked>()?, start)
            }
        };
        let start = usize::from(packed.offset()) + slice_start;
        start.is_multiple_of(FL_CHUNK_SIZE).then_some(Self {
            packed,
            first_block: start / FL_CHUNK_SIZE,
        })
    }
}

/// Performs the low-level delta decompression on primitive values.
///
/// All chunks must be full 1024-element chunks (deltas length must be a multiple of 1024).
pub(crate) fn decompress_primitive<T, const LANES: usize>(bases: &[T], deltas: &[T]) -> Buffer<T>
where
    T: NativePType + Delta + Transpose,
{
    let (chunks, remainder) = deltas.as_chunks::<1024>();
    debug_assert!(
        remainder.is_empty(),
        "deltas must be padded to a multiple of 1024"
    );
    assert_eq!(bases.len(), chunks.len() * LANES);

    // Allocate a result array.
    let mut output = BufferMut::with_capacity(deltas.len());
    // Bound to the requested length: `spare_capacity_mut` may expose extra over-aligned capacity
    // beyond `deltas.len()`, which would desync the `zip_eq` with `chunks` below and panic.
    let (output_chunks, _) = output.spare_capacity_mut()[..deltas.len()].as_chunks_mut::<1024>();

    // Loop over all the chunks
    let mut transposed: [T; 1024] = [T::default(); 1024];
    for ((i, chunk), output_chunk) in chunks.iter().enumerate().zip_eq(output_chunks.iter_mut()) {
        Delta::undelta::<LANES>(
            chunk,
            unsafe { &*(bases[i * LANES..(i + 1) * LANES].as_ptr().cast()) },
            &mut transposed,
        );

        untranspose(&transposed, output_chunk);
    }

    unsafe { output.set_len(deltas.len()) };

    output.freeze()
}

/// Delta decompression of bit-packed deltas, unpacking and prefix-summing each chunk in one pass.
///
/// `BitPacking::unpack` and `Delta::undelta` visit a chunk in the same lane order, so the fused
/// kernel produces the same transposed chunk as unpacking followed by undelta. The chunk still has
/// to be untransposed into logical order, exactly like the unfused path.
///
/// A patch changes every later value of its lane, so a chunk with patches is unpacked, patched and
/// then prefix-summed. Chunks without patches take the fused kernel.
pub(crate) fn fused_decompress_primitive<T, const LANES: usize>(
    bases: &[T],
    deltas: PackedDeltas<'_>,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>>
where
    T: NativePType + PhysicalPType<Physical = T> + UndeltaPack,
{
    debug_assert!(
        len.is_multiple_of(FL_CHUNK_SIZE),
        "deltas must be padded to a multiple of 1024"
    );
    let num_chunks = len / FL_CHUNK_SIZE;
    assert_eq!(bases.len(), num_chunks * LANES);
    let bp = deltas.packed;
    let blocks = deltas.first_block..deltas.first_block + num_chunks;
    let patches = chunk_patches::<T>(bp, &blocks, ctx)?;
    let mut patches = patches.as_slice();

    let mut output = BufferMut::<T>::with_capacity(len);
    let (output_chunks, _) = output.spare_capacity_mut()[..len].as_chunks_mut::<FL_CHUNK_SIZE>();

    let mut decoded = 0;
    let mut transposed: [T; FL_CHUNK_SIZE] = [T::default(); FL_CHUNK_SIZE];
    let mut unpacked: [T; FL_CHUNK_SIZE] = [T::default(); FL_CHUNK_SIZE];
    let mut decode_block = |packed: &[T], bit_width: usize, block: usize| {
        if !blocks.contains(&block) {
            return;
        }
        let chunk = block - blocks.start;
        let base = &bases[chunk * LANES..][..LANES];
        let patched = patches.partition_point(|&(index, _)| index < (chunk + 1) * FL_CHUNK_SIZE);
        let (chunk_patches, rest) = patches.split_at(patched);
        patches = rest;
        if chunk_patches.is_empty() {
            T::undelta_pack_width(bit_width, packed, base, &mut transposed);
        } else {
            // SAFETY: the block walk bounds `packed` to one block at `bit_width`.
            unsafe { BitPacking::unchecked_unpack(bit_width, packed, &mut unpacked) };
            for &(index, value) in chunk_patches {
                unpacked[index % FL_CHUNK_SIZE] = value;
            }
            let base = base
                .first_chunk::<LANES>()
                .vortex_expect("one base per lane");
            Delta::undelta::<LANES>(&unpacked, base, &mut transposed);
        }
        untranspose(&transposed, &mut output_chunks[chunk]);
        decoded += 1;
    };

    let packed = bp.packed_slice::<T>();
    let offset = usize::from(bp.offset());
    match bp.bit_widths() {
        BitWidthsView::Global(bit_width) => {
            let bit_width = usize::from(bit_width);
            for_each_packed_chunk::<T, _>(packed, bit_width, offset, bp.len(), |packed, range| {
                decode_block(packed, bit_width, range.start / FL_CHUNK_SIZE)
            })?;
        }
        BitWidthsView::Blocked(offsets) => {
            // The validated offsets bound every block to its bit width within `packed`.
            let offsets = execute_block_offsets(bp, offsets, ctx)?;
            for_each_packed_block::<T, _>(
                packed,
                &offsets,
                offset,
                bp.len(),
                |packed, bit_width, range| {
                    decode_block(packed, bit_width, range.start / FL_CHUNK_SIZE)
                },
            );
        }
    }
    vortex_ensure_eq!(decoded, num_chunks, "bit-packed deltas hold too few blocks");

    // SAFETY: every chunk of `output` was initialized above.
    unsafe { output.set_len(len) };

    Ok(output.freeze())
}

/// The patches of `bp` that fall within `blocks`, as positions counted from the start of the
/// first block, in ascending order.
fn chunk_patches<T: NativePType>(
    bp: ArrayView<'_, BitPacked>,
    blocks: &Range<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<(usize, T)>> {
    let Some(patches) = bp.patches() else {
        return Ok(Vec::new());
    };
    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches
        .values()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(T::PTYPE);
    let values = values.as_slice::<T>();
    let start = blocks.start * FL_CHUNK_SIZE;
    let positions = start..blocks.end * FL_CHUNK_SIZE;
    // Patch indices count from the start of `bp`, which sits `offset` into its first block.
    let shift = usize::from(bp.offset());
    let mut chunk_patches: Vec<(usize, T)> =
        match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
            indices
                .as_slice::<P>()
                .iter()
                .zip_eq(values)
                .filter_map(|(&index, &value)| {
                    let position = AsPrimitive::<usize>::as_(index) - patches.offset() + shift;
                    positions
                        .contains(&position)
                        .then(|| (position - start, value))
                })
                .collect()
        });
    chunk_patches.sort_unstable_by_key(|&(position, _)| position);
    Ok(chunk_patches)
}

/// Reorder a transposed chunk into logical order, like [`Transpose::untranspose`].
///
/// Writes the output in order and gathers from the input, which the compiler vectorizes far
/// better than the scattered writes of `Transpose::untranspose`.
#[inline]
pub(crate) fn untranspose<T: Copy>(
    input: &[T; FL_CHUNK_SIZE],
    output: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
) {
    // `transpose` maps logical index `row * 128 + order * 16 + lane` to transposed index
    // `lane * 64 + FL_ORDER[order] * 8 + row`, and FL_ORDER is its own inverse.
    for lane in 0..16 {
        for (fl_order, &order) in FL_ORDER.iter().enumerate() {
            for row in 0..8 {
                output[lane * 64 + fl_order * 8 + row].write(input[row * 128 + order * 16 + lane]);
            }
        }
    }
}

/// Runtime bit width dispatch for [`Delta::undelta_pack`].
pub(crate) trait UndeltaPack: Delta {
    /// Unpack one chunk of `bit_width`-bit deltas from `packed` and prefix-sum each lane starting
    /// from its value in `base`, writing the chunk in transposed order to `output`.
    ///
    /// # Panics
    ///
    /// Panics if `bit_width` is wider than `Self`, `packed` does not hold exactly one chunk at
    /// `bit_width`, or `base` does not hold one value per lane.
    fn undelta_pack_width(
        bit_width: usize,
        packed: &[Self],
        base: &[Self],
        output: &mut [Self; FL_CHUNK_SIZE],
    );
}

macro_rules! impl_undelta_pack {
    ($T:ty, [$($W:literal),*]) => {
        impl UndeltaPack for $T {
            #[inline]
            fn undelta_pack_width(
                bit_width: usize,
                packed: &[Self],
                base: &[Self],
                output: &mut [Self; FL_CHUNK_SIZE],
            ) {
                const LANES: usize = <$T as FastLanes>::LANES;
                let base: &[Self; LANES] = base.try_into().unwrap_or_else(|_| {
                    panic!("expected {LANES} bases, got {}", base.len())
                });
                assert_eq!(
                    packed.len(),
                    FL_CHUNK_SIZE * bit_width / <$T as FastLanes>::T,
                    "packed chunk length does not match bit width {bit_width}"
                );
                match bit_width {
                    $($W => {
                        const B: usize = FL_CHUNK_SIZE * $W / <$T as FastLanes>::T;
                        // SAFETY: the length of `packed` was checked against `B` above.
                        let packed = unsafe { &*packed.as_ptr().cast::<[Self; B]>() };
                        <$T as Delta>::undelta_pack::<LANES, $W, B>(packed, base, output);
                    })*
                    _ => panic!("unsupported bit width {bit_width} for {}", stringify!($T)),
                }
            }
        }
    };
}

impl_undelta_pack!(u8, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_undelta_pack!(
    u16,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
);
impl_undelta_pack!(
    u32,
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    ]
);
impl_undelta_pack!(
    u64,
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    ]
);
