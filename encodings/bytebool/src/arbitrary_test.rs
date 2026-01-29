// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;
    use vortex_array::Array;
    use vortex_dtype::Nullability;

    use crate::ArbitraryByteBoolArray;

    #[test]
    fn test_arbitrary_bytebool_basic() {
        let seed: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate arbitrary ByteBoolArray
        let bytebool = ArbitraryByteBoolArray::arbitrary(&mut u).unwrap();
        let array = &bytebool.0;

        // Verify basic properties
        assert!(array.len() <= 100, "Length should be bounded");
        assert!(array.dtype().is_boolean(), "ByteBool dtype should be bool");

        println!(
            "ByteBoolArray: len={}, dtype={}, nullability={:?}",
            array.len(),
            array.dtype(),
            array.dtype().nullability()
        );

        // Check some values
        if array.len() > 0 {
            let slice = array.as_slice();
            println!("  First few values: {:?}", &slice[..slice.len().min(5)]);
        }
    }

    #[test]
    fn test_arbitrary_bytebool_non_nullable() {
        let seed: Vec<u8> = (100..3100).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate non-nullable ByteBoolArray
        let bytebool =
            ArbitraryByteBoolArray::with_nullability(&mut u, Nullability::NonNullable, Some(10))
                .unwrap();
        let array = &bytebool.0;

        assert_eq!(array.len(), 10);
        assert_eq!(array.dtype().nullability(), Nullability::NonNullable);

        println!(
            "ByteBoolArray (non-nullable): len={}, dtype={}",
            array.len(),
            array.dtype()
        );

        // All values should be valid
        for i in 0..array.len() {
            assert!(array.is_valid(i).unwrap(), "All values should be valid");
        }

        let slice = array.as_slice();
        println!("  Values: {:?}", slice);
    }

    #[test]
    fn test_arbitrary_bytebool_nullable() {
        let seed: Vec<u8> = (200..3200).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate nullable ByteBoolArray
        let bytebool =
            ArbitraryByteBoolArray::with_nullability(&mut u, Nullability::Nullable, Some(20))
                .unwrap();
        let array = &bytebool.0;

        assert_eq!(array.len(), 20);
        assert_eq!(array.dtype().nullability(), Nullability::Nullable);

        println!(
            "ByteBoolArray (nullable): len={}, dtype={}",
            array.len(),
            array.dtype()
        );

        // Count valid values
        let valid_count = (0..array.len())
            .filter(|&i| array.is_valid(i).unwrap())
            .count();
        println!(
            "  Valid: {}/{}, values: {:?}",
            valid_count,
            array.len(),
            array.as_slice()
        );
    }
}
