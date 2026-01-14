// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrayref::array_mut_ref;
use arrayref::array_ref;
use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use num_traits::WrappingSub;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;

pub fn delta_compress(array: &PrimitiveArray) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    // TODO(ngates): fill forward nulls?
    // let filled = fill_forward(array)?.to_primitive()?;

    // Compress the filled array
    let (bases, deltas) = match_each_unsigned_integer_ptype!(array.ptype(), |T| {
        const LANES: usize = T::LANES;
        let (bases, deltas) = compress_primitive::<T, LANES>(array.as_slice::<T>());
        (
            // To preserve nullability, we include Validity
            PrimitiveArray::new(bases, array.dtype().nullability().into()),
            PrimitiveArray::new(deltas, array.validity().clone()),
        )
    });

    Ok((bases, deltas))
}

fn compress_primitive<T: NativePType + Delta + Transpose + WrappingSub, const LANES: usize>(
    array: &[T],
) -> (Buffer<T>, Buffer<T>) {
    // How many fastlanes vectors we will process.
    let num_chunks = array.len() / 1024;

    // Allocate result arrays.
    let mut bases = BufferMut::with_capacity(num_chunks * T::LANES + 1);
    let mut deltas = BufferMut::with_capacity(array.len());

    // Loop over all the 1024-element chunks.
    if num_chunks > 0 {
        let mut transposed: [T; 1024] = [T::default(); 1024];

        for i in 0..num_chunks {
            let start_elem = i * 1024;
            let chunk: &[T; 1024] = array_ref![array, start_elem, 1024];
            Transpose::transpose(chunk, &mut transposed);

            // Initialize and store the base vector for each chunk
            bases.extend_from_slice(&transposed[0..T::LANES]);

            deltas.reserve(1024);
            let delta_len = deltas.len();
            unsafe {
                deltas.set_len(delta_len + 1024);
                Delta::delta::<LANES>(
                    &transposed,
                    &*(transposed[0..T::LANES].as_ptr().cast()),
                    array_mut_ref![deltas[delta_len..], 0, 1024],
                );
            }
        }
    }

    // To avoid padding, the remainder is encoded with scalar logic.
    let remainder_size = array.len() % 1024;
    if remainder_size > 0 {
        let chunk = &array[array.len() - remainder_size..];
        let mut base_scalar = chunk[0];
        bases.push(base_scalar);
        for next in chunk {
            let diff = next.wrapping_sub(&base_scalar);
            deltas.push(diff);
            base_scalar = *next;
        }
    }

    assert_eq!(
        bases.len(),
        num_chunks * T::LANES + (if remainder_size > 0 { 1 } else { 0 })
    );
    assert_eq!(deltas.len(), array.len());

    (bases.freeze(), deltas.freeze())
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use crate::DeltaArray;
    use crate::delta::array::delta_decompress::delta_decompress;

    #[test]
    fn test_compress() {
        do_roundtrip_test((0u32..10_000).collect());
    }

    #[test]
    fn test_compress_nullable() {
        do_roundtrip_test(PrimitiveArray::from_option_iter(
            (0u32..10_000).map(|i| (i % 2 == 0).then_some(i)),
        ));
    }

    #[test]
    fn test_compress_overflow() {
        do_roundtrip_test((0..10_000).map(|i| (i % (u8::MAX as i32)) as u8).collect());
    }

    fn do_roundtrip_test(input: PrimitiveArray) {
        let delta = DeltaArray::try_from_primitive_array(&input).unwrap();
        assert_eq!(delta.len(), input.len());
        let decompressed = delta_decompress(&delta);
        assert_arrays_eq!(decompressed, input);
    }
}
