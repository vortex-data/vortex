// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for decimal scalar casting functionality.

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, DecimalDType, Nullability, PType};

    use crate::{DecimalValue, Scalar};

    #[test]
    fn test_decimal_to_primitive_i32() {
        // Create a decimal with value 42.50 (scale=2, so internal value is 4250)
        let decimal = Scalar::decimal(
            DecimalValue::I32(4250),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Cast to i32 - should truncate to 42
        let result = decimal.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_ok());
        let i32_scalar = result.unwrap();
        assert_eq!(i32_scalar.as_primitive().typed_value::<i32>().unwrap(), 42);
    }

    #[test]
    fn test_decimal_to_primitive_f64() {
        // Create a decimal with value 123.45 (scale=2, so internal value is 12345)
        let decimal = Scalar::decimal(
            DecimalValue::I32(12345),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Cast to f64 - should preserve decimal value
        let result = decimal.cast(&DType::Primitive(PType::F64, Nullability::NonNullable));
        assert!(result.is_ok());
        let f64_scalar = result.unwrap();
        assert_eq!(f64_scalar.as_primitive().typed_value::<f64>().unwrap(), 123.45);
    }

    #[test]
    fn test_decimal_to_primitive_f32() {
        // Create a decimal with value 99.99 (scale=2, so internal value is 9999)
        let decimal = Scalar::decimal(
            DecimalValue::I16(9999),
            DecimalDType::new(4, 2),
            Nullability::NonNullable,
        );

        // Cast to f32
        let result = decimal.cast(&DType::Primitive(PType::F32, Nullability::NonNullable));
        assert!(result.is_ok());
        let f32_scalar = result.unwrap();
        assert!((f32_scalar.as_primitive().typed_value::<f32>().unwrap() - 99.99).abs() < 0.01);
    }

    #[test]
    fn test_decimal_to_primitive_u8_overflow() {
        // Create a decimal with value 256.00 (scale=2, so internal value is 25600)
        let decimal = Scalar::decimal(
            DecimalValue::I32(25600),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Cast to u8 - should fail due to overflow
        let result = decimal.cast(&DType::Primitive(PType::U8, Nullability::NonNullable));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of range for u8"));
    }

    #[test]
    fn test_decimal_to_decimal_same_type() {
        // Create a decimal with specific precision and scale
        let decimal = Scalar::decimal(
            DecimalValue::I64(123456),
            DecimalDType::new(10, 3),
            Nullability::NonNullable,
        );

        // Cast to same decimal type but nullable
        let target_dtype = DType::Decimal(DecimalDType::new(10, 3), Nullability::Nullable);
        let result = decimal.cast(&target_dtype);
        assert!(result.is_ok());
        
        let casted = result.unwrap();
        assert_eq!(casted.dtype(), &target_dtype);
        assert_eq!(
            casted.as_decimal().decimal_value(),
            &Some(DecimalValue::I64(123456))
        );
    }

    #[test]
    fn test_decimal_to_decimal_different_scale() {
        // Create a decimal with scale=2
        let decimal = Scalar::decimal(
            DecimalValue::I32(10000), // Represents 100.00
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Cast to decimal with scale=4
        // TODO: This should properly rescale, but for now it preserves the raw value
        let target_dtype = DType::Decimal(DecimalDType::new(10, 4), Nullability::NonNullable);
        let result = decimal.cast(&target_dtype);
        assert!(result.is_ok());
        
        let casted = result.unwrap();
        assert_eq!(
            casted.as_decimal().decimal_value(),
            &Some(DecimalValue::I32(10000))
        );
    }

    #[test]
    fn test_null_decimal_cast() {
        // Create a null decimal
        let null_decimal = Scalar::null(DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::Nullable,
        ));

        // Cast to i32 - should produce null i32
        let result = null_decimal.cast(&DType::Primitive(PType::I32, Nullability::Nullable));
        assert!(result.is_ok());
        let i32_scalar = result.unwrap();
        assert!(i32_scalar.is_null());
        assert_eq!(i32_scalar.dtype(), &DType::Primitive(PType::I32, Nullability::Nullable));
    }

    #[test]
    fn test_decimal_i256_to_primitive() {
        // Create a decimal with i256 value
        use crate::i256;
        let large_value = i256::from_i128(1234567890);
        let decimal = Scalar::decimal(
            DecimalValue::I256(large_value),
            DecimalDType::new(20, 6), // scale=6
            Nullability::NonNullable,
        );

        // Cast to f64 - value is 1234.567890
        let result = decimal.cast(&DType::Primitive(PType::F64, Nullability::NonNullable));
        assert!(result.is_ok());
        let f64_scalar = result.unwrap();
        assert!((f64_scalar.as_primitive().typed_value::<f64>().unwrap() - 1234.567890).abs() < 0.000001);
    }

    #[test]
    fn test_decimal_negative_value_cast() {
        // Create a negative decimal value
        let decimal = Scalar::decimal(
            DecimalValue::I32(-5000), // Represents -50.00 with scale=2
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Cast to i32
        let result = decimal.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_ok());
        let i32_scalar = result.unwrap();
        assert_eq!(i32_scalar.as_primitive().typed_value::<i32>().unwrap(), -50);

        // Cast to u32 - should fail due to negative value
        let result = decimal.cast(&DType::Primitive(PType::U32, Nullability::NonNullable));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of range for u32"));
    }

    #[test]
    fn test_decimal_cast_preserves_nullability() {
        // Non-nullable decimal
        let decimal = Scalar::decimal(
            DecimalValue::I32(100),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Cast to nullable i32
        let result = decimal.cast(&DType::Primitive(PType::I32, Nullability::Nullable));
        assert!(result.is_ok());
        let nullable_scalar = result.unwrap();
        assert_eq!(nullable_scalar.dtype().nullability(), Nullability::Nullable);
        assert!(!nullable_scalar.is_null());
    }

    #[test]
    fn test_decimal_to_unsupported_type() {
        let decimal = Scalar::decimal(
            DecimalValue::I32(100),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        // Try to cast to string - should fail
        let result = decimal.cast(&DType::Utf8(Nullability::NonNullable));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Can't cast decimal scalar to"));
    }

    #[test] 
    fn test_decimal_i8_all_primitive_casts() {
        // Test casting from smallest decimal type to all primitive types
        let decimal = Scalar::decimal(
            DecimalValue::I8(50), // Represents 5.0 with scale=1
            DecimalDType::new(3, 1),
            Nullability::NonNullable,
        );

        // Cast to each primitive type
        let casts = vec![
            (PType::U8, 5u64),
            (PType::U16, 5),
            (PType::U32, 5),
            (PType::U64, 5),
            (PType::I8, 5),
            (PType::I16, 5),
            (PType::I32, 5),
            (PType::I64, 5),
        ];

        for (ptype, expected) in casts {
            let result = decimal.cast(&DType::Primitive(ptype, Nullability::NonNullable));
            assert!(result.is_ok(), "Failed to cast to {:?}", ptype);
            let scalar = result.unwrap();
            
            // Check the value matches expected
            match ptype {
                PType::U8 => assert_eq!(scalar.as_primitive().typed_value::<u8>().unwrap() as u64, expected),
                PType::U16 => assert_eq!(scalar.as_primitive().typed_value::<u16>().unwrap() as u64, expected),
                PType::U32 => assert_eq!(scalar.as_primitive().typed_value::<u32>().unwrap() as u64, expected),
                PType::U64 => assert_eq!(scalar.as_primitive().typed_value::<u64>().unwrap(), expected),
                PType::I8 => assert_eq!(scalar.as_primitive().typed_value::<i8>().unwrap() as u64, expected),
                PType::I16 => assert_eq!(scalar.as_primitive().typed_value::<i16>().unwrap() as u64, expected),
                PType::I32 => assert_eq!(scalar.as_primitive().typed_value::<i32>().unwrap() as u64, expected),
                PType::I64 => assert_eq!(scalar.as_primitive().typed_value::<i64>().unwrap() as u64, expected),
                _ => panic!("Unexpected type"),
            }
        }
    }
}