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

use crate::DeltaArray;
use crate::bit_transpose::untranspose_validity;
use crate::delta::array::DeltaArrayExt;

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
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
