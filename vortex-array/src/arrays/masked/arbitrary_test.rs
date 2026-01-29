// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::Array;
    use crate::arrays::ArbitraryMaskedArray;

    #[test]
    fn test_arbitrary_masked_basic() {
        let seed: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate arbitrary MaskedArray
        let masked = ArbitraryMaskedArray::arbitrary(&mut u).unwrap();
        let array = &masked.0;

        // Verify basic properties
        assert!(array.len() <= 100, "Length should be bounded");
        // MaskedArray always produces nullable output
        assert_eq!(
            array.dtype().nullability(),
            Nullability::Nullable,
            "MaskedArray always produces nullable dtype"
        );

        println!("MaskedArray: len={}, dtype={}", array.len(), array.dtype());

        // Count valid values
        let valid_count = (0..array.len())
            .filter(|&i| array.is_valid(i).unwrap())
            .count();
        println!("  Valid: {}/{}", valid_count, array.len());
    }

    #[test]
    fn test_arbitrary_masked_with_dtype() {
        let seed: Vec<u8> = (100..3100).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate MaskedArray wrapping i32
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let masked = ArbitraryMaskedArray::with_dtype(&mut u, &dtype, Some(15)).unwrap();
        let array = &masked.0;

        assert_eq!(array.len(), 15);
        assert_eq!(array.dtype().as_ptype(), PType::I32);
        assert_eq!(array.dtype().nullability(), Nullability::Nullable);

        println!(
            "MaskedArray (I32): len={}, dtype={}",
            array.len(),
            array.dtype()
        );

        // Count valid values
        let valid_count = (0..array.len())
            .filter(|&i| array.is_valid(i).unwrap())
            .count();
        println!("  Valid: {}/{}", valid_count, array.len());
    }

    #[test]
    fn test_arbitrary_masked_child_is_nonnullable() {
        let seed: Vec<u8> = (200..3200).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate MaskedArray - the child should always have all valid values
        let dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        let masked = ArbitraryMaskedArray::with_dtype(&mut u, &dtype, Some(10)).unwrap();
        let array = &masked.0;

        // The child array should be non-nullable (all valid)
        let child = array.child();
        assert!(
            child.all_valid().unwrap(),
            "MaskedArray's child must have all valid values"
        );

        println!(
            "MaskedArray child: len={}, dtype={}, all_valid={}",
            child.len(),
            child.dtype(),
            child.all_valid().unwrap()
        );
    }
}
