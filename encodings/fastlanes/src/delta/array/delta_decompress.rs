// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrayref::array_mut_ref;
use arrayref::array_ref;
use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use num_traits::WrappingAdd;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::DeltaArray;

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let deltas = array.deltas().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    // TODO(connor): This is incorrect, we need to untranspose the validity!!!

    let validity =
        Validity::from_mask(array.deltas().validity_mask()?, array.dtype().nullability());
    let validity = validity.slice(start..end)?;

    Ok(match_each_unsigned_integer_ptype!(deltas.ptype(), |T| {
        const LANES: usize = T::LANES;

        let buffer = decompress_primitive::<T, LANES>(bases.as_slice(), deltas.as_slice());
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    }))
}

// TODO(ngates): can we re-use the deltas buffer for the result? Might be tricky given the
//  traversal ordering, but possibly doable.
/// Performs the low-level delta decompression on primitive values.
pub(crate) fn decompress_primitive<T, const LANES: usize>(bases: &[T], deltas: &[T]) -> Buffer<T>
where
    T: NativePType + Delta + Transpose + WrappingAdd,
{
    // How many fastlanes vectors we will process.
    let num_chunks = deltas.len() / 1024;

    // Allocate a result array.
    let mut output = BufferMut::with_capacity(deltas.len());

    // Loop over all the chunks
    if num_chunks > 0 {
        let mut transposed: [T; 1024] = [T::default(); 1024];

        for i in 0..num_chunks {
            let start_elem = i * 1024;
            let chunk: &[T; 1024] = array_ref![deltas, start_elem, 1024];

            // Initialize the base vector for this chunk
            Delta::undelta::<LANES>(
                chunk,
                unsafe { &*(bases[i * LANES..(i + 1) * LANES].as_ptr().cast()) },
                &mut transposed,
            );

            let output_len = output.len();
            unsafe { output.set_len(output_len + 1024) }
            Transpose::untranspose(&transposed, array_mut_ref![output[output_len..], 0, 1024]);
        }
    }
    assert_eq!(output.len() % 1024, 0);

    // The remainder was encoded with scalar logic, so we need to scalar decode it.
    let remainder_size = deltas.len() % 1024;
    if remainder_size > 0 {
        let chunk = &deltas[num_chunks * 1024..];
        assert_eq!(bases.len(), num_chunks * LANES + 1);
        let mut base_scalar = bases[num_chunks * LANES];
        for next_diff in chunk {
            let next = next_diff.wrapping_add(&base_scalar);
            output.push(next);
            base_scalar = next;
        }
    }

    output.freeze()
}
