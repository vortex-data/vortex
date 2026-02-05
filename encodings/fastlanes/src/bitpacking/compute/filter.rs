// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Filter kernel has been migrated to vtable/kernels/filter.rs

#[cfg(test)]
mod test {
    use vortex_array::Array;
    use vortex_array::IntoArray as _;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = bitpacked.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18])
        );
    }

    #[test]
    fn take_sliced_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = sliced.filter(mask).unwrap();
        assert_arrays_eq!(primitive_result, PrimitiveArray::from_iter([31u8, 33]));
    }

    #[test]
    fn filter_bitpacked() {
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(4096, (0..1024).collect()))
            .unwrap();
        assert_arrays_eq!(
            filtered.to_primitive(),
            PrimitiveArray::from_iter((0..1024).map(|i| (i % 63) as u8))
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 9).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), (0..250).collect()))
            .unwrap()
            .to_primitive();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values[0..250].iter().copied())
        );
    }

    #[test]
    fn test_filter_bitpacked_conformance() {
        // Test with u8 values
        let unpacked = buffer![1u8, 2, 3, 4, 5].into_array();
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 3).unwrap();
        test_filter_conformance(bitpacked.as_ref());

        // Test with u32 values
        let unpacked = buffer![100u32, 200, 300, 400, 500].into_array();
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 9).unwrap();
        test_filter_conformance(bitpacked.as_ref());

        // Test with nullable values
        let unpacked = PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]);
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 3).unwrap();
        test_filter_conformance(bitpacked.as_ref());
    }

    /// Regression test for signed integers with patches.
    ///
    /// When filtering signed integers that have patches (exceptions), the patches
    /// are stored with the signed type but FastLanes uses unsigned types internally.
    /// This test ensures that the type handling is correct.
    #[test]
    fn filter_bitpacked_signed_with_patches() {
        // Create signed integer values where some exceed the bit width (causing patches).
        // Values 0-127 fit in 7 bits, but 1000 and 2000 do not.
        let values: Vec<i32> = vec![0, 10, 1000, 20, 30, 2000, 40, 50, 60, 70];
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 7).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Filter to include some patched and some non-patched values.
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), vec![0, 2, 5, 9]))
            .unwrap()
            .to_primitive();

        assert_arrays_eq!(filtered, PrimitiveArray::from_iter([0i32, 1000, 2000, 70]));
    }

    /// Regression test for signed integers with patches using low selectivity.
    ///
    /// This test uses a low selectivity filter which takes a different code path
    /// that doesn't fully decompress the array first.
    #[test]
    fn filter_bitpacked_signed_with_patches_low_selectivity() {
        // Create a larger array with signed integers and some patches.
        let values: Vec<i32> = (0..1000)
            .map(|i| {
                if i % 100 == 0 {
                    10000 + i // These will be patches (exceed 7 bits)
                } else {
                    i % 128 // These fit in 7 bits
                }
            })
            .collect();
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 7).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Use low selectivity (only select 2% of values) to avoid full decompression.
        let indices: Vec<usize> = (0..20).collect();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), indices))
            .unwrap()
            .to_primitive();

        let expected: Vec<i32> = values[0..20].to_vec();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected));
    }
}
