// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use itertools::Itertools;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::DeltaArray;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    let validity = array.validity()?;

    // Deltas bit-packed whole, with no patches, unpack and undelta in one fused pass per chunk.
    if let Some(bp) = array.deltas().as_opt::<BitPacked>()
        && bp.offset() == 0
        && bp.patches().is_none()
        && bp.len() % 1024 == 0
    {
        let original_ptype = bp.dtype().as_ptype();
        let width = bp.bit_width() as usize;
        let bases = bases.reinterpret_cast(original_ptype.to_unsigned());
        let decoded = match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
            let buffer = decompress_packed::<T, { T::LANES }>(
                bases.as_slice(),
                bp.packed_slice::<T>(),
                width,
                bp.len() / 1024,
            );
            PrimitiveArray::new(buffer.slice(start..end), validity)
        });
        return Ok(decoded.reinterpret_cast(original_ptype));
    }

    let deltas = array.deltas().clone().execute::<PrimitiveArray>(ctx)?;

    let original_ptype = deltas.ptype();
    // Signed inputs are processed through their unsigned counterpart; `wrapping_add` on the
    // raw bytes inverts the `wrapping_sub` done at compress time regardless of signedness.
    let bases = bases.reinterpret_cast(original_ptype.to_unsigned());
    let deltas = deltas.reinterpret_cast(original_ptype.to_unsigned());

    let decoded = match_each_unsigned_integer_ptype!(deltas.ptype(), |T| {
        const LANES: usize = T::LANES;

        let buffer = decompress_primitive::<T, LANES>(bases.as_slice(), deltas.as_slice());
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    });

    Ok(decoded.reinterpret_cast(original_ptype))
}

/// Performs the low-level delta decompression on primitive values.
///
/// All chunks must be full 1024-element chunks (deltas length must be a multiple of 1024).
/// Unpack each chunk's deltas from `width` bits and undelta them in the same pass, then untranspose.
fn decompress_packed<T, const LANES: usize>(
    bases: &[T],
    packed: &[T],
    width: usize,
    num_chunks: usize,
) -> Buffer<T>
where
    T: NativePType + UndeltaPack + Transpose,
{
    assert!(bases.len() >= num_chunks * LANES);
    let words = 1024 * width / (8 * size_of::<T>());
    assert!(packed.len() >= num_chunks * words);
    let mut output = BufferMut::with_capacity(num_chunks * 1024);
    let (output_chunks, _) =
        output.spare_capacity_mut()[..num_chunks * 1024].as_chunks_mut::<1024>();
    let mut transposed: [T; 1024] = [T::default(); 1024];
    for (i, output_chunk) in output_chunks.iter_mut().enumerate() {
        T::undelta_pack_width(
            width,
            &packed[i * words..(i + 1) * words],
            &bases[i * LANES..(i + 1) * LANES],
            &mut transposed,
        );
        Transpose::untranspose(&transposed, unsafe {
            mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(output_chunk)
        });
    }
    // SAFETY: every chunk was written above.
    unsafe { output.set_len(num_chunks * 1024) };
    output.freeze()
}

/// FastLanes' fused unpack-and-undelta, with the bit width chosen at runtime.
trait UndeltaPack: Sized {
    /// `input` holds one chunk packed at `width` bits, and `base` one base per lane.
    fn undelta_pack_width(width: usize, input: &[Self], base: &[Self], output: &mut [Self; 1024]);
}

macro_rules! impl_undelta_pack {
    ($t:ty, [$($w:literal),*]) => {
        impl UndeltaPack for $t {
            fn undelta_pack_width(
                width: usize,
                input: &[Self],
                base: &[Self],
                output: &mut [Self; 1024],
            ) {
                const LANES: usize = <$t as FastLanes>::LANES;
                let base: &[Self; LANES] = base.try_into().expect("one base per lane");
                match width {
                    $($w => {
                        const B: usize = 1024 * $w / (8 * size_of::<$t>());
                        let input: &[Self; B] = input.try_into().expect("one packed chunk");
                        Delta::undelta_pack::<LANES, $w, B>(input, base, output);
                    })*
                    _ => unreachable!("bit width {width} exceeds {}", stringify!($t)),
                }
            }
        }
    };
}

impl_undelta_pack!(u8, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_undelta_pack!(u16, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
impl_undelta_pack!(u32, [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32
]);
impl_undelta_pack!(u64, [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
]);

pub(crate) fn decompress_primitive<T, const LANES: usize>(bases: &[T], deltas: &[T]) -> Buffer<T>
where
    T: NativePType + Delta + Transpose,
{
    let (chunks, remainder) = deltas.as_chunks::<1024>();
    debug_assert!(
        remainder.is_empty(),
        "deltas must be padded to a multiple of 1024"
    );
    // Use >= because cross-type casts (e.g. u32→u64) may produce more bases than the
    // target LANES requires. Only the first chunks.len() * LANES bases are used.
    assert!(bases.len() >= chunks.len() * LANES);

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

        Transpose::untranspose(&transposed, unsafe {
            mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(output_chunk)
        });
    }

    unsafe { output.set_len(deltas.len()) };

    output.freeze()
}
