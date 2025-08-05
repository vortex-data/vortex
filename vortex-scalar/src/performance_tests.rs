// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for performance optimizations in scalar types.

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    
    use crate::{BinaryScalar, Scalar, Utf8Scalar};

    #[test]
    fn test_utf8_value_ref_avoids_clone() {
        let large_string = "x".repeat(10000);
        let scalar = Scalar::utf8(large_string.clone(), Nullability::NonNullable);
        let utf8_scalar = Utf8Scalar::try_from(&scalar).unwrap();
        
        // value_ref() returns a reference without cloning
        let value_ref = utf8_scalar.value_ref();
        assert!(value_ref.is_some());
        assert_eq!(value_ref.unwrap().as_str(), &large_string);
        
        // value() still works and returns an owned copy
        let value_owned = utf8_scalar.value();
        assert!(value_owned.is_some());
        assert_eq!(value_owned.unwrap().as_str(), &large_string);
    }
    
    #[test]
    fn test_binary_value_ref_avoids_clone() {
        let large_binary = vec![42u8; 10000];
        let scalar = Scalar::binary(large_binary.clone(), Nullability::NonNullable);
        let binary_scalar = BinaryScalar::try_from(&scalar).unwrap();
        
        // value_ref() returns a reference without cloning
        let value_ref = binary_scalar.value_ref();
        assert!(value_ref.is_some());
        assert_eq!(value_ref.unwrap().as_slice(), &large_binary);
        
        // value() still works and returns an owned copy
        let value_owned = binary_scalar.value();
        assert!(value_owned.is_some());
        assert_eq!(value_owned.unwrap().as_slice(), &large_binary);
    }
    
    #[test]
    fn test_null_scalar_value_ref() {
        // Test that null scalars return None for both value() and value_ref()
        let null_utf8 = Scalar::null_typed::<String>();
        let utf8_scalar = Utf8Scalar::try_from(&null_utf8).unwrap();
        assert!(utf8_scalar.value_ref().is_none());
        assert!(utf8_scalar.value().is_none());
        
        let null_binary = Scalar::null(vortex_dtype::DType::Binary(Nullability::Nullable));
        let binary_scalar = BinaryScalar::try_from(&null_binary).unwrap();
        assert!(binary_scalar.value_ref().is_none());
        assert!(binary_scalar.value().is_none());
    }
}