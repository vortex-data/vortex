// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::bit_transpose::transpose_validity;

pub fn delta_compress(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let (bases, deltas) = match_each_unsigned_integer_ptype!(array.ptype(), |T| {
        let (bases, deltas) = compress_primitive::<T, { T::LANES }>(array.as_slice::<T>());
        // TODO(robert): This can be avoided if we add TransposedBoolArray that performs index translation when necessary.
        let validity = transpose_validity(array.validity(), ctx)?;
        (
            PrimitiveArray::new(bases, array.dtype().nullability().into()),
            PrimitiveArray::new(deltas, validity),
        )
    });

    Ok((bases, deltas))
}

fn compress_primitive<T: NativePType + Delta + Transpose, const LANES: usize>(
    array: &[T],
) -> (Buffer<T>, Buffer<T>) {
    let padded_len = array.len().next_multiple_of(1024);
    let num_chunks = padded_len / 1024;
    let bases_len = num_chunks * LANES;

    // Split into full 1024-element chunks and a remainder.
    let (full_chunks, remainder) = array.as_chunks::<1024>();

    // Allocate result arrays.
    let mut bases = BufferMut::with_capacity(bases_len);
    let mut deltas = BufferMut::with_capacity(padded_len);
    let (output_deltas, _) = deltas.spare_capacity_mut().as_chunks_mut::<1024>();

    // Loop over all full 1024-element chunks.
    let mut transposed: [T; 1024] = [T::default(); 1024];
    for (chunk, output) in full_chunks.iter().zip(output_deltas.iter_mut()) {
        Transpose::transpose(chunk, &mut transposed);
        bases.extend_from_slice(&transposed[0..T::LANES]);

        unsafe {
            Delta::delta::<LANES>(
                &transposed,
                &*(transposed[0..T::LANES].as_ptr().cast()),
                mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(output),
            );
        }
    }

    // Pad the remainder to 1024 elements and process as a full chunk.
    if !remainder.is_empty() {
        let mut padded_chunk = [T::default(); 1024];
        padded_chunk[..remainder.len()].copy_from_slice(remainder);

        Transpose::transpose(&padded_chunk, &mut transposed);
        bases.extend_from_slice(&transposed[0..T::LANES]);

        unsafe {
            Delta::delta::<LANES>(
                &transposed,
                &*(transposed[0..T::LANES].as_ptr().cast()),
                mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(
                    &mut output_deltas[full_chunks.len()],
                ),
            );
        }
    }

    unsafe { deltas.set_len(padded_len) };

    assert_eq!(bases.len(), bases_len);
    assert_eq!(deltas.len(), padded_len);

    (bases.freeze(), deltas.freeze())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DeltaArray;
    use crate::delta::array::delta_decompress::delta_decompress;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[rstest]
    #[case((0u32..10_000).collect())]
    #[case((0..10_000).map(|i| (i % (u8::MAX as i32)) as u8).collect())]
    #[case(PrimitiveArray::from_option_iter(
            (0u32..10_000).map(|i| (i % 2 == 0).then_some(i)),
    ))]
    fn test_compress(#[case] array: PrimitiveArray) -> VortexResult<()> {
        let delta =
            DeltaArray::try_from_primitive_array(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(delta.len(), array.len());
        let decompressed = delta_decompress(&delta, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(decompressed, array);
        Ok(())
    }
}
