use arrayref::{array_mut_ref, array_ref};
use fastlanes::{Delta, Transpose};
use num_traits::{WrappingAdd, WrappingSub};
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::slice;
use vortex_array::compute::unary::fill_forward;
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayLen, IntoArrayVariant};
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType, Nullability};
use vortex_error::VortexResult;

use crate::DeltaArray;

pub fn delta_compress(array: &PrimitiveArray) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    // Fill forward nulls
    let filled = fill_forward(array.as_ref())?.into_primitive()?;

    // Compress the filled array
    let (bases, deltas) = match_each_unsigned_integer_ptype!(array.ptype(), |$T| {
        let (bases, deltas) = compress_primitive(filled.maybe_null_slice::<$T>());
        let base_validity = (array.validity().nullability() != Nullability::NonNullable)
            .then(|| Validity::AllValid)
            .unwrap_or(Validity::NonNullable);
        let delta_validity = (array.validity().nullability() != Nullability::NonNullable)
            .then(|| Validity::AllValid)
            .unwrap_or(Validity::NonNullable);
        (
            // To preserve nullability, we include Validity
            PrimitiveArray::from_vec(bases, base_validity),
            PrimitiveArray::from_vec(deltas, delta_validity),
        )
    });

    Ok((bases, deltas))
}

fn compress_primitive<T: NativePType + Delta + Transpose + WrappingSub>(
    array: &[T],
) -> (Vec<T>, Vec<T>)
where
    [(); T::LANES]:,
{
    // How many fastlanes vectors we will process.
    let num_chunks = array.len() / 1024;

    // Allocate result arrays.
    let mut bases = Vec::with_capacity(num_chunks * T::LANES + 1);
    let mut deltas = Vec::with_capacity(array.len());

    // Loop over all the 1024-element chunks.
    if num_chunks > 0 {
        let mut transposed: [T; 1024] = [T::default(); 1024];
        let mut base = [T::default(); T::LANES];

        for i in 0..num_chunks {
            let start_elem = i * 1024;
            let chunk: &[T; 1024] = array_ref![array, start_elem, 1024];
            Transpose::transpose(chunk, &mut transposed);

            // Initialize and store the base vector for each chunk
            // TODO(ngates): avoid copying the base vector
            base.copy_from_slice(&transposed[0..T::LANES]);
            bases.extend(base);

            deltas.reserve(1024);
            let delta_len = deltas.len();
            unsafe {
                deltas.set_len(delta_len + 1024);
                Delta::delta(
                    &transposed,
                    &base,
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

    (bases, deltas)
}

pub fn delta_decompress(array: DeltaArray) -> VortexResult<PrimitiveArray> {
    let bases = array.bases().into_primitive()?;
    let deltas = array.deltas().into_primitive()?;
    let decoded = match_each_unsigned_integer_ptype!(deltas.ptype(), |$T| {
        PrimitiveArray::from_vec(
            decompress_primitive::<$T>(bases.maybe_null_slice(), deltas.maybe_null_slice()),
            array.validity()
        )
    });

    slice(decoded, array.offset(), array.offset() + array.len())?.into_primitive()
}

fn decompress_primitive<T: NativePType + Delta + Transpose + WrappingAdd>(
    bases: &[T],
    deltas: &[T],
) -> Vec<T>
where
    [(); T::LANES]:,
{
    // How many fastlanes vectors we will process.
    let num_chunks = deltas.len() / 1024;

    // How long each base vector will be.
    let lanes = T::LANES;

    // Allocate a result array.
    let mut output = Vec::with_capacity(deltas.len());

    // Loop over all the chunks
    if num_chunks > 0 {
        let mut transposed: [T; 1024] = [T::default(); 1024];
        let mut base = [T::default(); T::LANES];

        for i in 0..num_chunks {
            let start_elem = i * 1024;
            let chunk: &[T; 1024] = array_ref![deltas, start_elem, 1024];

            // Initialize the base vector for this chunk
            // TODO(ngates): avoid copying the bases
            base.copy_from_slice(&bases[i * lanes..(i + 1) * lanes]);
            Delta::undelta(chunk, &base, &mut transposed);

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
        assert_eq!(bases.len(), num_chunks * lanes + 1);
        let mut base_scalar = bases[num_chunks * lanes];
        for next_diff in chunk {
            let next = next_diff.wrapping_add(&base_scalar);
            output.push(next);
            base_scalar = next;
        }
    }

    output
}

#[cfg(test)]
mod test {
    use vortex_array::ArrayLen;

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
        let decompressed = delta_decompress(delta).unwrap();
        let decompressed_slice = decompressed.maybe_null_slice::<T>();
        assert_eq!(decompressed_slice.len(), input.len());
        for (actual, expected) in decompressed_slice.iter().zip(input) {
            assert_eq!(actual, &expected);
        }
    }
}
