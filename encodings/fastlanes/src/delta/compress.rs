// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrayref::{array_mut_ref, array_ref};
use fastlanes::{Delta, FastLanes, Transpose};
use num_traits::{WrappingAdd, WrappingSub};
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{NativePType, Nullability, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;

use crate::DeltaArray;

pub fn delta_compress(array: &PrimitiveArray) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    // TODO(ngates): fill forward nulls?
    // let filled = fill_forward(array)?.to_primitive()?;

    // Compress the filled array
    let (bases, deltas) = match_each_unsigned_integer_ptype!(array.ptype(), |T| {
        const LANES: usize = T::LANES;
        let (bases, deltas) = compress_primitive::<T, LANES>(array.as_slice::<T>());
        let (base_validity, delta_validity) =
            if array.validity().nullability() != Nullability::NonNullable {
                (Validity::AllValid, Validity::AllValid)
            } else {
                (Validity::NonNullable, Validity::NonNullable)
            };
        (
            // To preserve nullability, we include Validity
            PrimitiveArray::new(bases, base_validity),
            PrimitiveArray::new(deltas, delta_validity),
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

pub fn delta_decompress(array: &DeltaArray) -> VortexResult<PrimitiveArray> {
    let bases = array.bases().to_primitive()?;
    let deltas = array.deltas().to_primitive()?;
    let decoded = match_each_unsigned_integer_ptype!(deltas.ptype(), |T| {
        const LANES: usize = T::LANES;

        PrimitiveArray::new(
            decompress_primitive::<T, LANES>(bases.as_slice(), deltas.as_slice()),
            array.validity().clone(),
        )
    });

    decoded
        .slice(array.offset(), array.offset() + array.len())
        .to_primitive()
}

// TODO(ngates): can we re-use the deltas buffer for the result? Might be tricky given the
//  traversal ordering, but possibly doable.
fn decompress_primitive<T: NativePType + Delta + Transpose + WrappingAdd, const LANES: usize>(
    bases: &[T],
    deltas: &[T],
) -> Buffer<T> {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_compress() {
        do_roundtrip_test((0u32..10_000).collect::<Vec<_>>());
    }

    #[test]
    fn test_compress_overflow() {
        do_roundtrip_test(
            (0..10_000)
                .map(|i| (i % (u8::MAX as i32)) as u8)
                .collect::<Vec<_>>(),
        );
    }

    fn do_roundtrip_test<T: NativePType>(input: Vec<T>) {
        let delta = DeltaArray::try_from_vec(input.clone()).unwrap();
        assert_eq!(delta.len(), input.len());
        let decompressed = delta_decompress(&delta).unwrap();
        let decompressed_slice = decompressed.as_slice::<T>();
        assert_eq!(decompressed_slice.len(), input.len());
        for (actual, expected) in decompressed_slice.iter().zip(input) {
            assert_eq!(actual, &expected);
        }
    }
}
