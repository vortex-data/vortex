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
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::bit_transpose::transpose_validity;
use crate::fill_forward_nulls;
pub fn delta_compress(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let validity = array.validity()?;
    let (bases, deltas) = match_each_unsigned_integer_ptype!(array.ptype(), |T| {
        // Fill-forward null values so that transposed deltas at null positions remain
        // small. Without this, bitpacking may skip patches for null positions, and the
        // corrupted delta values propagate through the cumulative sum during decompression.
        let filled = fill_forward_nulls(array.to_buffer::<T>(), &validity, ctx)?;
        let (bases, deltas) = compress_primitive::<T, { T::LANES }>(&filled);
        // TODO(robert): This can be avoided if we add TransposedBoolArray that performs index translation when necessary.
        let validity = transpose_validity(&validity, ctx)?;
        (
            PrimitiveArray::new(bases, array.dtype().nullability().into()),
            PrimitiveArray::new(deltas, validity),
        )
    });

    Ok((bases, deltas))
}

fn compress_primitive<T, const LANES: usize>(array: &[T]) -> (Buffer<T>, Buffer<T>)
where
    T: NativePType + Delta + Transpose,
{
    let padded_len = array.len().next_multiple_of(FL_CHUNK_SIZE);
    let bases_len = (padded_len / FL_CHUNK_SIZE) * LANES;

    // Split into full 1024-element chunks and a remainder.
    let (full_chunks, remainder) = array.as_chunks::<FL_CHUNK_SIZE>();

    // Allocate result arrays.
    let mut bases = BufferMut::with_capacity(bases_len);
    let mut deltas = BufferMut::with_capacity(padded_len);
    let (output_deltas, _) = deltas.spare_capacity_mut().as_chunks_mut::<FL_CHUNK_SIZE>();

    // Loop over all full 1024-element chunks.
    let mut transposed: [T; FL_CHUNK_SIZE] = [T::default(); FL_CHUNK_SIZE];
    let mut process_chunk = |input: &[T; FL_CHUNK_SIZE], output: &mut [MaybeUninit<T>; 1024]| {
        Transpose::transpose(input, &mut transposed);
        bases.extend_from_slice(&transposed[0..T::LANES]);

        unsafe {
            Delta::delta::<LANES>(
                &transposed,
                &*(transposed[0..T::LANES].as_ptr().cast()),
                mem::transmute::<&mut [MaybeUninit<T>; FL_CHUNK_SIZE], &mut [T; FL_CHUNK_SIZE]>(
                    output,
                ),
            );
        }
    };
    for (chunk, output) in full_chunks.iter().zip(output_deltas.iter_mut()) {
        process_chunk(chunk, output);
    }

    // Pad the remainder to 1024 elements and process as a full chunk.
    if !remainder.is_empty() {
        let mut padded_chunk = [T::default(); FL_CHUNK_SIZE];
        padded_chunk[..remainder.len()].copy_from_slice(remainder);
        process_chunk(&padded_chunk, &mut output_deltas[full_chunks.len()]);
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;
    use crate::bitpack_compress::bitpack_encode;
    use crate::delta::array::delta_decompress::delta_decompress;
    use crate::delta_compress;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[rstest]
    #[case((0u32..10_000).collect())]
    #[case((0..10_000).map(|i| (i % (u8::MAX as i32)) as u8).collect())]
    #[case(PrimitiveArray::from_option_iter(
            (0u32..10_000).map(|i| (i % 2 == 0).then_some(i)),
    ))]
    fn test_compress(#[case] array: PrimitiveArray) -> VortexResult<()> {
        let delta = Delta::try_from_primitive_array(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(delta.len(), array.len());
        let decompressed = delta_decompress(&delta, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(decompressed, array);
        Ok(())
    }

    /// Regression test: delta + bitpacked encoding must correctly round-trip nullable arrays
    /// where null positions contain arbitrary values. Without fill-forward, the delta cumulative
    /// sum propagates corrupted values from null positions.
    #[test]
    fn delta_bitpacked_trailing_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter(
            (0u8..200).map(|i| (!(50..100).contains(&i)).then_some(i)),
        );
        let (bases, deltas) = delta_compress(&array, &mut ctx).unwrap();
        let bitpacked_deltas = bitpack_encode(&deltas, 1, None, &mut ctx).unwrap();
        let packed_delta = Delta::try_new(
            bases.into_array(),
            bitpacked_deltas.into_array(),
            0,
            array.len(),
        )
        .vortex_expect("Delta array construction should succeed");
        let packed_delta_prim = packed_delta
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(packed_delta_prim, array);
        Ok(())
    }
}
