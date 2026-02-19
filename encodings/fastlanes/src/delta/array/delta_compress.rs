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
    use std::sync::LazyLock;

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

    #[test]
    fn test_compress() -> VortexResult<()> {
        do_roundtrip_test((0u32..10_000).collect())
    }

    #[test]
    fn test_compress_nullable() -> VortexResult<()> {
        do_roundtrip_test(PrimitiveArray::from_option_iter(
            (0u32..10_000).map(|i| (i % 2 == 0).then_some(i)),
        ))
    }

    #[test]
    fn test_compress_overflow() -> VortexResult<()> {
        do_roundtrip_test((0..10_000).map(|i| (i % (u8::MAX as i32)) as u8).collect())
    }

    fn do_roundtrip_test(input: PrimitiveArray) -> VortexResult<()> {
        let delta = DeltaArray::try_from_primitive_array(&input)?;
        assert_eq!(delta.len(), input.len());
        let decompressed = delta_decompress(&delta, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(decompressed, input);
        Ok(())
    }

    // --- Delta + BitPacking stacked roundtrip tests ---
    //
    // These tests delta-compress a PrimitiveArray, then bitpack the deltas child,
    // reassemble a DeltaArray with the BitPackedArray as its deltas child, and
    // verify the full decompress roundtrip.

    use vortex_array::IntoArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    use crate::bitpack_compress::bitpack_to_best_bit_width;

    /// Roundtrip helper: delta compress → bitpack deltas → reassemble → decompress.
    /// Uses `crate::test::SESSION` which registers `BitPackedVTable` so the
    /// execution context can decompress the stacked encoding.
    fn do_delta_bitpacked_roundtrip_test(input: PrimitiveArray) -> VortexResult<()> {
        let (bases, deltas) = super::delta_compress(&input)?;
        let bitpacked_deltas = bitpack_to_best_bit_width(&deltas)?;
        let delta = DeltaArray::try_from_delta_compress_parts(
            bases.into_array(),
            bitpacked_deltas.into_array(),
        )?;
        assert_eq!(delta.len(), input.len());
        let decompressed =
            delta_decompress(&delta, &mut crate::test::SESSION.create_execution_ctx())?;
        assert_arrays_eq!(decompressed, input);
        Ok(())
    }

    #[test]
    fn test_delta_bitpacked_non_nullable() -> VortexResult<()> {
        // 10_000 elements: multiple 1024-element SIMD chunks + scalar remainder.
        do_delta_bitpacked_roundtrip_test((0u32..10_000).collect())
    }

    #[test]
    fn test_delta_bitpacked_nullable_multi_chunk() -> VortexResult<()> {
        // Monotonic buffer ensures small deltas so bitpacking is effective.
        // Every 3rd value is null.
        let values: Buffer<u32> = (0u32..10_000).collect();
        let validity = Validity::from_iter((0u32..10_000).map(|i| i % 3 != 0));
        do_delta_bitpacked_roundtrip_test(PrimitiveArray::new(values, validity))
    }

    #[test]
    fn test_delta_bitpacked_nullable_small() -> VortexResult<()> {
        // Fewer than 1024 elements: only the scalar delta path is exercised.
        let values: Buffer<u32> = (0u32..500).collect();
        let validity = Validity::from_iter((0u32..500).map(|i| i % 5 != 0));
        do_delta_bitpacked_roundtrip_test(PrimitiveArray::new(values, validity))
    }

    #[test]
    fn test_delta_bitpacked_nullable_exact_chunk() -> VortexResult<()> {
        // Exactly 1024 elements: one full SIMD chunk, no remainder.
        let values: Buffer<u32> = (0u32..1024).collect();
        let validity = Validity::from_iter((0u32..1024).map(|i| i % 4 != 0));
        do_delta_bitpacked_roundtrip_test(PrimitiveArray::new(values, validity))
    }

    #[test]
    fn test_delta_bitpacked_nullable_sparse_nulls() -> VortexResult<()> {
        // 2048 elements (two full SIMD chunks) with nulls at specific boundary positions.
        let n = 2048u32;
        let values: Buffer<u32> = (0..n).collect();
        let validity = Validity::from_iter(
            (0..n).map(|i| i != 0 && i != 512 && i != 1023 && i != 1024 && i != 1500),
        );
        do_delta_bitpacked_roundtrip_test(PrimitiveArray::new(values, validity))
    }

    // --- scalar_at / validity_mask tests for nullable DeltaArrays ---
    //
    // These test that individual element access and the validity mask are correct
    // despite the internal mismatch between transposed delta data and
    // original-order validity in the deltas child. If validity ever gets
    // incorrectly transposed, these will catch it.

    use vortex_array::Array;
    use vortex_mask::Mask;

    /// Build a nullable DeltaArray (1024 u32 values, one full SIMD chunk).
    /// Nulls at positions given by `null_positions`. The underlying buffer is
    /// monotonically increasing so delta values are small.
    fn make_nullable_delta(null_positions: &[u32]) -> VortexResult<(DeltaArray, Vec<bool>)> {
        let n = 1024u32;
        let valid: Vec<bool> = (0..n).map(|i| !null_positions.contains(&i)).collect();
        let values: Buffer<u32> = (0..n).collect();
        let validity = Validity::from_iter(valid.iter().copied());
        let input = PrimitiveArray::new(values, validity);
        Ok((DeltaArray::try_from_primitive_array(&input)?, valid))
    }

    /// Same as `make_nullable_delta` but bitpacks the deltas child.
    fn make_nullable_delta_bitpacked(
        null_positions: &[u32],
    ) -> VortexResult<(DeltaArray, Vec<bool>)> {
        let n = 1024u32;
        let valid: Vec<bool> = (0..n).map(|i| !null_positions.contains(&i)).collect();
        let values: Buffer<u32> = (0..n).collect();
        let validity = Validity::from_iter(valid.iter().copied());
        let input = PrimitiveArray::new(values, validity);

        let (bases, deltas) = super::delta_compress(&input)?;
        let bitpacked = bitpack_to_best_bit_width(&deltas)?;
        let delta =
            DeltaArray::try_from_delta_compress_parts(bases.into_array(), bitpacked.into_array())?;
        Ok((delta, valid))
    }

    #[test]
    fn test_scalar_at_nullable_delta() -> VortexResult<()> {
        // Nulls at positions that would map to different transposed positions
        // for u32 (32 lanes). Position 1 and position 32 are in different
        // transposed locations; if validity were transposed, checking these
        // positions would give wrong results.
        let null_positions = [1, 31, 512, 1023];
        let (delta, valid) = make_nullable_delta(&null_positions)?;
        let arr = delta.into_array();

        for &pos in &null_positions {
            let scalar = arr.scalar_at(pos as usize)?;
            assert!(scalar.is_null(), "expected null at position {pos}");
        }

        // Check several non-null positions that are "near" the null positions
        // in both original and transposed order.
        for pos in [0u32, 2, 30, 32, 33, 64, 500, 513, 1000, 1022] {
            assert!(valid[pos as usize], "precondition: position {pos} is valid");
            let scalar = arr.scalar_at(pos as usize)?;
            assert!(!scalar.is_null(), "expected non-null at position {pos}");
        }
        Ok(())
    }

    #[test]
    fn test_scalar_at_nullable_delta_bitpacked() -> VortexResult<()> {
        let null_positions = [1, 31, 512, 1023];
        let (delta, valid) = make_nullable_delta_bitpacked(&null_positions)?;
        let arr = delta.into_array();

        for &pos in &null_positions {
            let scalar = arr.scalar_at(pos as usize)?;
            assert!(scalar.is_null(), "expected null at position {pos}");
        }

        for pos in [0u32, 2, 30, 32, 33, 64, 500, 513, 1000, 1022] {
            assert!(valid[pos as usize]);
            let scalar = arr.scalar_at(pos as usize)?;
            assert!(!scalar.is_null(), "expected non-null at position {pos}");
        }
        Ok(())
    }

    #[test]
    fn test_validity_mask_nullable_delta() -> VortexResult<()> {
        let null_positions = [1, 31, 32, 512, 1023];
        let (delta, valid) = make_nullable_delta(&null_positions)?;
        let mask = delta.into_array().validity_mask()?;

        for i in 0..1024usize {
            let expected = valid[i];
            let actual = mask.value(i);
            assert_eq!(actual, expected, "validity mismatch at position {i}");
        }
        Ok(())
    }

    #[test]
    fn test_validity_mask_nullable_delta_bitpacked() -> VortexResult<()> {
        let null_positions = [1, 31, 32, 512, 1023];
        let (delta, valid) = make_nullable_delta_bitpacked(&null_positions)?;
        let mask = delta.into_array().validity_mask()?;

        for i in 0..1024usize {
            let expected = valid[i];
            let actual = mask.value(i);
            assert_eq!(actual, expected, "validity mismatch at position {i}");
        }
        Ok(())
    }

    #[test]
    fn test_scalar_at_sliced_nullable_delta() -> VortexResult<()> {
        // Create 2048 elements (2 SIMD chunks) with some nulls, then slice
        // across the chunk boundary and verify scalar_at on the slice.
        let n = 2048u32;
        let null_positions = [1, 500, 1023, 1024, 1500, 2047];
        let valid: Vec<bool> = (0..n).map(|i| !null_positions.contains(&i)).collect();
        let values: Buffer<u32> = (0..n).collect();
        let validity = Validity::from_iter(valid.iter().copied());
        let input = PrimitiveArray::new(values, validity);

        let delta = DeltaArray::try_from_primitive_array(&input)?;
        // Slice across chunk boundary: positions 500..1500 of original.
        let sliced = delta.slice(500..1500)?;
        assert_eq!(sliced.len(), 1000);

        // Original position 500 is null → sliced position 0 is null.
        assert!(sliced.scalar_at(0)?.is_null());
        // Original position 1023 is null → sliced position 523 is null.
        assert!(sliced.scalar_at(523)?.is_null());
        // Original position 1024 is null → sliced position 524 is null.
        assert!(sliced.scalar_at(524)?.is_null());

        // Original position 501 is valid → sliced position 1 is valid.
        assert!(!sliced.scalar_at(1)?.is_null());
        // Original position 1022 is valid → sliced position 522 is valid.
        assert!(!sliced.scalar_at(522)?.is_null());
        // Original position 1499 is valid → sliced position 999 is valid.
        assert!(!sliced.scalar_at(999)?.is_null());
        Ok(())
    }

    #[test]
    fn test_validity_mask_sliced_nullable_delta() -> VortexResult<()> {
        let n = 2048u32;
        let null_positions = [0, 1, 31, 32, 500, 1023, 1024, 1500, 2047];
        let valid: Vec<bool> = (0..n).map(|i| !null_positions.contains(&i)).collect();
        let values: Buffer<u32> = (0..n).collect();
        let validity = Validity::from_iter(valid.iter().copied());
        let input = PrimitiveArray::new(values, validity);

        let delta = DeltaArray::try_from_primitive_array(&input)?;
        let sliced = delta.slice(500..1500)?;
        let mask: Mask = sliced.validity_mask()?;

        for i in 0..1000usize {
            let orig_pos = 500 + i;
            let expected = valid[orig_pos];
            let actual = mask.value(i);
            assert_eq!(
                actual, expected,
                "validity mismatch at slice pos {i} (original pos {orig_pos})"
            );
        }
        Ok(())
    }
}
