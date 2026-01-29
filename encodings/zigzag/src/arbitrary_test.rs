// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;
    use vortex_array::ToCanonical;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::ArbitraryZigZagArray;

    #[test]
    fn test_arbitrary_zigzag_basic() {
        let seed: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate arbitrary ZigZagArray
        let zigzag = ArbitraryZigZagArray::arbitrary(&mut u).unwrap();
        let array = &zigzag.0;

        // Verify basic properties
        assert!(array.len() <= 100, "Length should be bounded");
        // ZigZag's dtype is the SIGNED type (decoded logical type)
        // The internal encoded child is unsigned
        assert!(
            array.dtype().is_signed_int(),
            "ZigZag dtype is signed (decoded type), got: {}",
            array.dtype()
        );

        println!("ZigZagArray: len={}, dtype={}", array.len(), array.dtype());

        // Verify we can decode it
        if array.len() > 0 {
            let primitive = array.to_primitive();
            let slice = primitive.as_slice::<i8>();
            println!("  First few values: {:?}", &slice[..slice.len().min(5)]);
        }
    }

    #[test]
    fn test_arbitrary_zigzag_with_ptype() {
        let seed: Vec<u8> = (100..3100).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Generate with specific ptype - U32 encoded => I32 logical dtype
        let zigzag = ArbitraryZigZagArray::with_ptype(
            &mut u,
            PType::U32,
            Nullability::NonNullable,
            Some(10),
        )
        .unwrap();
        let array = &zigzag.0;

        assert_eq!(array.len(), 10);
        // ZigZag converts unsigned encoded to signed dtype
        assert_eq!(array.dtype().as_ptype(), PType::I32);
        assert_eq!(array.dtype().nullability(), Nullability::NonNullable);

        println!(
            "ZigZagArray (encoded U32 -> I32): len={}, dtype={}",
            array.len(),
            array.dtype()
        );

        // Verify we can decode
        let primitive = array.to_primitive();
        let slice = primitive.as_slice::<i32>();
        println!("  Values: {:?}", slice);
    }
}
