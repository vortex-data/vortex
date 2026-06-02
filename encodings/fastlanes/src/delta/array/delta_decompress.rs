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
#[cfg(feature = "unstable_encodings")]
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DeltaArray;
use crate::bit_transpose::untranspose_validity;
use crate::delta::array::DeltaArrayExt;
#[cfg(feature = "unstable_encodings")]
use crate::{BitPacked, BitPackedArrayExt, FoR, r#for::FoRArrayExt};

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    // Fast path: a fully fused `delta(for(bitpacking))` decode that unpacks, applies the
    // frame-of-reference, and inverts the delta encoding in a single pass over the packed buffer.
    #[cfg(feature = "unstable_encodings")]
    if let Some(decoded) = try_fused_for_bitpacking(array, ctx)? {
        return Ok(decoded);
    }

    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let deltas = array.deltas().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    let validity = untranspose_validity(&deltas.validity()?, ctx)?;
    let validity = validity.slice(start..end)?;

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

/// Attempts the fused `delta(for(bitpacking))` decode.
///
/// Returns `Some` when the `deltas` child is a [`FoR`] array with an unsigned reference wrapping a
/// [`BitPacked`] array stored as full, zero-offset chunks with no patches. In that case the packed
/// deltas are unpacked, FoR-decoded, and un-delta'd in a single pass via
/// [`Delta::unchecked_unfor_undelta_pack`]. Otherwise returns `None` so the caller falls back to the
/// generic path.
#[cfg(feature = "unstable_encodings")]
pub(crate) fn try_fused_for_bitpacking(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<PrimitiveArray>> {
    let Some(for_) = array.deltas().as_opt::<FoR>() else {
        return Ok(None);
    };
    // The fused kernel works in unsigned wrapping arithmetic; a signed reference would need a
    // bit-reinterpret that the generic path already handles correctly.
    if !for_.reference_scalar().dtype().is_unsigned_int() {
        return Ok(None);
    }
    let Some(bp) = for_.encoded().as_opt::<BitPacked>() else {
        return Ok(None);
    };
    // Patches and sliced (non-zero offset) bit-packed children are left to the generic path.
    if bp.patches().is_some() || bp.offset() != 0 {
        return Ok(None);
    }

    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    let validity = untranspose_validity(&bp.validity()?, ctx)?;
    let validity = validity.slice(start..end)?;

    let original_ptype = for_.ptype();
    let unsigned_ptype = original_ptype.to_unsigned();
    let bases = bases.reinterpret_cast(unsigned_ptype);

    let decoded = match_each_unsigned_integer_ptype!(unsigned_ptype, |T| {
        const LANES: usize = T::LANES;

        let reference = for_
            .reference_scalar()
            .as_primitive()
            .as_::<T>()
            .vortex_expect("FoR reference must be non-null and unsigned");
        let packed = bp.packed_slice::<T>();

        let buffer = decompress_fused::<T, LANES>(
            bases.as_slice(),
            packed,
            bp.bit_width() as usize,
            reference,
            bp.len(),
        );
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    });

    Ok(Some(decoded.reinterpret_cast(original_ptype)))
}

/// Fused low-level decode of bit-packed, FoR-encoded deltas.
///
/// `packed` holds `num_values / 1024` chunks each of `128 * bit_width / size_of::<T>()` packed
/// words. Each chunk is unpacked, FoR-decoded (wrapping-add `reference`) and un-delta'd in a single
/// pass, then untransposed back into logical order.
#[cfg(feature = "unstable_encodings")]
pub(crate) fn decompress_fused<T, const LANES: usize>(
    bases: &[T],
    packed: &[T],
    bit_width: usize,
    reference: T,
    num_values: usize,
) -> Buffer<T>
where
    T: NativePType + Delta + Transpose,
{
    debug_assert!(
        num_values.is_multiple_of(1024),
        "bit-packed deltas must be padded to a multiple of 1024"
    );
    let num_chunks = num_values / 1024;
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    debug_assert_eq!(packed.len(), num_chunks * elems_per_chunk);
    assert!(bases.len() >= num_chunks * LANES);

    let mut output = BufferMut::with_capacity(num_values);
    let (output_chunks, _) = output.spare_capacity_mut().as_chunks_mut::<1024>();

    let mut transposed: [T; 1024] = [T::default(); 1024];
    for (i, output_chunk) in output_chunks.iter_mut().enumerate() {
        let packed_chunk = &packed[i * elems_per_chunk..(i + 1) * elems_per_chunk];
        let base = &bases[i * LANES..(i + 1) * LANES];

        // SAFETY: `packed_chunk` has length `128 * bit_width / size_of::<T>()`, `base` has length
        // `LANES`, and `transposed` has length 1024, satisfying the kernel's contract.
        unsafe {
            Delta::unchecked_unfor_undelta_pack(
                bit_width,
                packed_chunk,
                reference,
                base,
                &mut transposed,
            );
        }

        Transpose::untranspose(&transposed, unsafe {
            mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(output_chunk)
        });
    }

    unsafe { output.set_len(num_values) };

    output.freeze()
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
    // Use >= because cross-type casts (e.g. u32→u64) may produce more bases than the
    // target LANES requires. Only the first chunks.len() * LANES bases are used.
    assert!(bases.len() >= chunks.len() * LANES);

    // Allocate a result array.
    let mut output = BufferMut::with_capacity(deltas.len());
    let (output_chunks, _) = output.spare_capacity_mut().as_chunks_mut::<1024>();

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
