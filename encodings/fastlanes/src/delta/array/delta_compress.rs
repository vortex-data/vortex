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

    // --- Demonstrates the internal data/validity mismatch in the deltas child ---
    //
    // The deltas child PrimitiveArray stores data in *transposed* (FastLanes)
    // order but validity in *original* order.  Any pushdown compute that
    // operates on the deltas child positionally (filter, take, compare) would
    // correlate the wrong data with the wrong validity.
    //
    // This test proves the mismatch exists by decompressing the deltas child
    // to a PrimitiveArray and showing that scalar_at on that PrimitiveArray
    // gives a DIFFERENT null/non-null answer than the DeltaArray's own
    // scalar_at for the same logical position.
    //
    // See also: https://github.com/vortex-data/vortex/pull/5048

    use vortex_array::ToCanonical;

    #[test]
    fn test_deltas_child_data_validity_mismatch() -> VortexResult<()> {
        // The deltas child PrimitiveArray stores delta values in transposed
        // (FastLanes lane) order, but validity in original order.  This means
        // scalar_at(i) on the raw child returns the transposed delta at
        // buffer position i but the null-flag for original position i.
        //
        // At non-null positions, the *values* from the deltas child will
        // differ from the fully-decompressed DeltaArray values because the
        // child holds transposed deltas while the DeltaArray decompresses
        // (undelta + untranspose) back to original values.
        //
        // This proves that any pushdown compute operating positionally on
        // the deltas child would associate the wrong value with the wrong
        // validity — the root cause behind:
        //   https://github.com/vortex-data/vortex/pull/5048
        let null_positions = [1u32, 100, 512, 900];
        let (delta, _valid) = make_nullable_delta(&null_positions)?;

        let deltas_child = delta.deltas().to_primitive();
        let delta_arr = delta.into_array();

        let mut value_mismatches = 0usize;
        for i in 0..1024usize {
            let decompressed = delta_arr.scalar_at(i)?;
            let from_child = deltas_child.scalar_at(i)?;

            // Null/non-null matches because both use the same
            // original-order validity.
            assert_eq!(
                decompressed.is_null(),
                from_child.is_null(),
                "null-flag unexpectedly differs at position {i}"
            );

            // But at non-null positions the VALUES should differ because
            // the child holds a transposed delta while the DeltaArray
            // returns the fully reconstructed original value.
            if !decompressed.is_null() {
                let v_decompressed: u32 = decompressed.as_primitive().typed_value().unwrap();
                let v_child: u32 = from_child.as_primitive().typed_value().unwrap();
                if v_decompressed != v_child {
                    value_mismatches += 1;
                }
            }
        }

        // The transposition reorders values within the 1024-element chunk,
        // so most non-null positions should show different values between
        // the raw child and the decompressed DeltaArray.
        assert!(
            value_mismatches > 0,
            "Expected value mismatches between deltas child (transposed) and \
             decompressed DeltaArray (original order), but found none — \
             this would mean the transposition is a no-op, which should not \
             happen for a non-trivial array"
        );
        Ok(())
    }

    /// Demonstrates the concrete validity/transpose mismatch by showing that
    /// specific positions in the deltas child have null flags that belong to
    /// different original elements than the data stored there.
    ///
    /// For u32 (32 lanes), transpose maps original positions to buffer positions
    /// via the FastLanes formula. E.g. `output[1] = input[transpose(1)]` where
    /// `transpose(1) = 64`. So buffer position 1 holds data from original
    /// position 64, but the validity bit at position 1 corresponds to original
    /// position 1.
    ///
    /// If original position 1 is null but position 64 is not, the deltas child
    /// at buffer position 1 has non-null data (from position 64) with a null
    /// validity flag (from position 1). Filtering non-null values from the
    /// child would incorrectly discard the data for original position 64.
    #[test]
    fn test_delta_nullable_transpose_validity_position_mismatch() -> VortexResult<()> {
        // Use the fastlanes transpose function to compute exact position mappings.
        // For u32: transpose(idx) gives the original-order position whose value
        // ends up at buffer position `idx` in the transposed output.
        //
        // transpose(1) = 64 for u32.
        // So buffer position 1 holds data from original position 64.
        // Make original position 1 null, original position 64 non-null.
        let n = 1024u32;
        let values: Buffer<u32> = (0..n).collect();
        let validity = Validity::from_iter((0..n).map(|i| {
            // null only at position 1
            i != 1
        }));
        let input = PrimitiveArray::new(values, validity);
        let delta = DeltaArray::try_from_primitive_array(&input)?;

        // The deltas child: data in transposed order, validity in original order.
        let deltas_child = delta.deltas().to_primitive();

        // Buffer position 1:
        //   - DATA comes from original position transpose(1) = 64 (non-null)
        //   - VALIDITY comes from original position 1 (null)
        // So the child reports position 1 as null, but the data there belongs
        // to a non-null original element (position 64).
        let child_scalar_1 = deltas_child.scalar_at(1)?;
        assert!(
            child_scalar_1.is_null(),
            "deltas child position 1 should report null (from original pos 1's validity), \
             even though the data there is from original position 64 (which is non-null)"
        );

        // The DeltaArray itself correctly reports position 1 as null.
        let delta_arr = delta.into_array();
        let delta_scalar_1 = delta_arr.scalar_at(1)?;
        assert!(
            delta_scalar_1.is_null(),
            "DeltaArray position 1 should be null (original position 1 was null)"
        );

        // Position 64 in the DeltaArray is non-null.
        let delta_scalar_64 = delta_arr.scalar_at(64)?;
        assert!(
            !delta_scalar_64.is_null(),
            "DeltaArray position 64 should be non-null"
        );
        let val_64: u32 = delta_scalar_64.as_primitive().typed_value().unwrap();
        assert_eq!(val_64, 64, "DeltaArray position 64 should have value 64");

        // Now demonstrate the harm: if we naively filter the deltas child to
        // keep only non-null positions, we'd drop buffer position 1 (which
        // holds the delta for original position 64). This is wrong—original
        // position 64 is valid and should be kept.
        //
        // Count how many positions have mismatched null status between the
        // transposed data's "true" source and the validity flag at that position.
        let child_validity_mask = deltas_child.validity_mask()?;
        let mut false_nulls = 0usize; // non-null data flagged as null
        let mut false_valids = 0usize; // null data flagged as valid

        // FL_ORDER is the 3-bit reversal permutation used by FastLanes.
        const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

        for buf_pos in 0..1024usize {
            // The FastLanes transpose formula: for output[i] = input[transpose(i)],
            // the data at buffer position buf_pos came from this original position.
            let lane = buf_pos % 16;
            let order = (buf_pos / 16) % 8;
            let row = buf_pos / 128;
            let source_original_pos = (lane * 64) + (FL_ORDER[order] * 8) + row;

            let source_is_null = source_original_pos == 1; // only position 1 is null
            let validity_says_null = !child_validity_mask.value(buf_pos);

            if source_is_null && !validity_says_null {
                false_valids += 1;
            }
            if !source_is_null && validity_says_null {
                false_nulls += 1;
            }
        }

        // Exactly 1 position has null data that the validity says is valid:
        // the buffer position where transpose(buf_pos) == 1, i.e., the position
        // that holds original position 1's data.
        assert_eq!(
            false_valids, 1,
            "expected exactly 1 null-data-flagged-as-valid (original pos 1's data \
             landed at some buffer position whose validity says 'valid')"
        );

        // Exactly 1 position has non-null data that the validity says is null:
        // buffer position 1, which holds data from original position 64 but
        // has original position 1's null flag.
        assert_eq!(
            false_nulls, 1,
            "expected exactly 1 non-null-data-flagged-as-null (buffer pos 1 holds \
             data from original pos 64 but validity from original pos 1)"
        );

        Ok(())
    }
}
