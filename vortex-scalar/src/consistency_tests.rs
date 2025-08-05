// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests demonstrating inconsistencies in the vortex-scalar crate.
//! These tests document current behavior that may be inconsistent or problematic.

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{BoolScalar, PrimitiveScalar, Scalar};

    // Demonstrates inconsistent null comparison behavior
    #[test]
    fn test_null_comparison_inconsistency() {
        // Test with primitive scalars
        let null_i32 = Scalar::null_typed::<i32>();
        let null_i64 = Scalar::null_typed::<i64>();
        
        let prim_i32 = PrimitiveScalar::try_from(&null_i32).unwrap();
        let prim_i64 = PrimitiveScalar::try_from(&null_i64).unwrap();
        
        // Primitive scalars check dtype compatibility first
        assert_eq!(prim_i32.partial_cmp(&prim_i64), None); // Different types => None
        
        // Test with boolean scalars
        let null_bool1 = Scalar::null(DType::Bool(Nullability::Nullable));
        let null_bool2 = Scalar::null(DType::Bool(Nullability::NonNullable));
        
        let bool1 = BoolScalar::try_from(&null_bool1).unwrap();
        let bool2 = BoolScalar::try_from(&null_bool2).unwrap();
        
        // Bool scalars compare values directly without checking dtype
        // This is inconsistent with primitive scalars
        assert!(bool1.partial_cmp(&bool2).is_some()); // Should be None for consistency
    }

    // Demonstrates that different scalar types have different Display formats
    #[test]
    fn test_display_format_inconsistency() {
        use std::fmt::Write;
        
        let mut output = String::new();
        
        // Primitive scalar shows just the value and type
        let prim = Scalar::primitive(42u32, Nullability::NonNullable);
        write!(&mut output, "{}", prim).unwrap();
        assert!(output.contains("42u32"));
        output.clear();
        
        // Bool scalar shows just the value
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);
        write!(&mut output, "{}", bool_scalar).unwrap();
        assert_eq!(output, "true");
        output.clear();
        
        // Decimal scalar shows value with precision/scale metadata
        use crate::{DecimalScalar, DecimalValue};
        use vortex_dtype::DecimalDType;
        
        let decimal = Scalar::decimal(
            DecimalValue::I32(4200),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        write!(&mut output, "{}", decimal).unwrap();
        // Decimal includes metadata in display
        assert!(output.contains("precision=10"));
        assert!(output.contains("scale=2"));
    }

    // Demonstrates missing error context in some scalar types
    #[test]
    fn test_error_message_inconsistency() {
        // Primitive scalar cast error has detailed context
        let prim = Scalar::primitive(42i32, Nullability::NonNullable);
        let result = prim.cast(&DType::Bool(Nullability::NonNullable));
        if let Err(e) = result {
            let error_str = format!("{}", e);
            // Primitive cast errors include source and target types
            assert!(error_str.contains("i32"));
            assert!(error_str.contains("bool"));
        }
        
        // Bool scalar cast error has minimal context
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);
        let result = bool_scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        if let Err(e) = result {
            let error_str = format!("{}", e);
            // Bool cast errors are less detailed
            assert!(error_str.contains("Can't cast bool"));
        }
    }

    // This used to panic with todo!() but now works correctly
    #[test]
    fn test_decimal_casting_now_works() {
        use crate::{DecimalValue};
        use vortex_dtype::DecimalDType;
        
        let decimal = Scalar::decimal(
            DecimalValue::I32(4200),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        
        // This used to panic with todo!() in lib.rs:231
        let result = decimal.cast(&DType::Primitive(PType::I64, Nullability::NonNullable));
        assert!(result.is_ok());
        let i64_scalar = result.unwrap();
        assert_eq!(i64_scalar.as_primitive().typed_value::<i64>().unwrap(), 42);
    }

    // Demonstrates that Option<T> handling is inconsistent between types
    #[test]
    fn test_option_handling_inconsistency() {
        // Primitive types have comprehensive Option<T> support
        let some_i32 = Scalar::primitive(42i32, Nullability::NonNullable);
        let none_i32 = Scalar::null_typed::<i32>();
        
        let extracted_some: Option<i32> = Option::try_from(&some_i32).unwrap();
        let extracted_none: Option<i32> = Option::try_from(&none_i32).unwrap();
        
        assert_eq!(extracted_some, Some(42));
        assert_eq!(extracted_none, None);
        
        // But decimal types don't have the same Option<T> TryFrom implementations
        // They use a different pattern with DecimalScalar as intermediary
    }

    // Test that demonstrates potential issues with typed null conversions
    #[test]
    fn test_typed_null_unit_conversion_surprising() {
        // This behavior is documented but potentially surprising
        let typed_null = Scalar::null_typed::<i32>();
        
        // A typed null (i32 null) successfully converts to unit type
        let unit_result = <()>::try_from(&typed_null);
        assert!(unit_result.is_ok()); // This might be unexpected!
        
        // But a non-null value correctly fails
        let non_null = Scalar::primitive(42i32, Nullability::NonNullable);
        let unit_result = <()>::try_from(&non_null);
        assert!(unit_result.is_err()); // Expected
    }

    // Demonstrates that equality checking doesn't always consider nullability
    #[test]
    fn test_nullability_in_equality() {
        let nullable = Scalar::primitive(42i32, Nullability::Nullable);
        let non_nullable = Scalar::primitive(42i32, Nullability::NonNullable);
        
        // These have different dtypes (different nullability)
        assert_ne!(nullable.dtype(), non_nullable.dtype());
        
        // But they compare as equal in value
        // This might be correct behavior but could be surprising
        assert_eq!(nullable, non_nullable);
    }
}