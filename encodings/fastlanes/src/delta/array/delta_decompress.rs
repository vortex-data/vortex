// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::Delta;
use fastlanes::FL_ORDER;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use itertools::Itertools;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

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

    let fused = array
        .deltas()
        .as_opt::<BitPacked>()
        .filter(|bp| is_fusable(*bp));

    let decoded = match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
        const LANES: usize = T::LANES;

        let buffer = match fused {
            Some(bp) => fused_decompress_primitive::<T, LANES>(bases.as_slice(), bp, ctx)?,
            None => {
                let deltas = array
                    .deltas()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?
                    .reinterpret_cast(original_ptype.to_unsigned());
                decompress_primitive::<T, LANES>(bases.as_slice(), deltas.as_slice())
            }
        };
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    });

    Ok(decoded.reinterpret_cast(original_ptype))
}

/// Whether bit-packed `deltas` can be decoded by the fused unpack and undelta kernel.
///
/// The fused kernel prefix-sums each lane while unpacking it, so every delta must come from the
/// packed buffer: a patch would change every later value of its lane. Delta slices its children
/// at chunk boundaries, so the packed chunks line up with the delta chunks when the offset is 0.
pub(crate) fn is_fusable(deltas: ArrayView<'_, BitPacked>) -> bool {
    deltas.offset() == 0 && deltas.patches().is_none()
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
/// `deltas` must pass [`is_fusable`].
pub(crate) fn fused_decompress_primitive<T, const LANES: usize>(
    bases: &[T],
    deltas: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>>
where
    T: NativePType + PhysicalPType<Physical = T> + UndeltaPack + Transpose,
{
    debug_assert!(is_fusable(deltas));
    let len = deltas.len();
    debug_assert!(
        len.is_multiple_of(FL_CHUNK_SIZE),
        "deltas must be padded to a multiple of 1024"
    );
    assert_eq!(bases.len(), len / FL_CHUNK_SIZE * LANES);

    let mut output = BufferMut::<T>::with_capacity(len);
    let (output_chunks, _) = output.spare_capacity_mut()[..len].as_chunks_mut::<FL_CHUNK_SIZE>();

    let mut transposed: [T; FL_CHUNK_SIZE] = [T::default(); FL_CHUNK_SIZE];
    let mut decode_chunk = |packed: &[T], bit_width: usize, range: Range<usize>| {
        debug_assert_eq!(range.len(), FL_CHUNK_SIZE);
        let chunk = range.start / FL_CHUNK_SIZE;
        T::undelta_pack_width(
            bit_width,
            packed,
            &bases[chunk * LANES..][..LANES],
            &mut transposed,
        );
        untranspose(&transposed, &mut output_chunks[chunk]);
    };

    let packed = deltas.packed_slice::<T>();
    match deltas.bit_widths() {
        BitWidthsView::Global(bit_width) => {
            let bit_width = usize::from(bit_width);
            for_each_packed_chunk::<T, _>(packed, bit_width, 0, len, |packed, range| {
                decode_chunk(packed, bit_width, range)
            })?;
        }
        BitWidthsView::Blocked(offsets) => {
            // The validated offsets bound every block to its bit width within `packed`.
            let offsets = execute_block_offsets(deltas, offsets, ctx)?;
            for_each_packed_block::<T, _>(packed, &offsets, 0, len, decode_chunk);
        }
    }

    // SAFETY: every chunk of `output` was initialized above.
    unsafe { output.set_len(len) };

    Ok(output.freeze())
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
