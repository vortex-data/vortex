// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for primitive scalar types, utility functions, and basic operations.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;
    use vortex_dtype::{DType, ExtDType, ExtID, Nullability, PType};
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

    #[test]
    fn default_value_for_complex_dtype() {
        let struct_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                (
                    "b",
                    DType::list(
                        DType::Primitive(PType::I8, Nullability::Nullable),
                        Nullability::NonNullable,
                    ),
                ),
                ("c", DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        let scalar = Scalar::default_value(struct_dtype.clone());
        assert_eq!(scalar.dtype(), &struct_dtype);

        let scalar = scalar.as_struct();

        let a_field = scalar.field("a").unwrap();
        assert_eq!(a_field.as_primitive().pvalue().unwrap(), PValue::I32(0));

        let b_field = scalar.field("b").unwrap();
        assert!(b_field.as_list().is_empty());

        let c_field = scalar.field("c").unwrap();
        assert!(c_field.is_null());
    }

    #[test]
    fn test_scalar_nbytes() {
        // Test null scalar - should be 0 bytes
        let null_scalar = Scalar::null(DType::Null);
        assert_eq!(null_scalar.nbytes(), 0);

        // Test bool scalar - should be 1 byte
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);
        assert_eq!(bool_scalar.nbytes(), 1);

        // Test primitive scalars
        let u8_scalar = Scalar::primitive(42u8, Nullability::NonNullable);
        assert_eq!(u8_scalar.nbytes(), 1);

        let u16_scalar = Scalar::primitive(1000u16, Nullability::NonNullable);
        assert_eq!(u16_scalar.nbytes(), 2);

        let u32_scalar = Scalar::primitive(100000u32, Nullability::NonNullable);
        assert_eq!(u32_scalar.nbytes(), 4);

        let u64_scalar = Scalar::primitive(10000000000u64, Nullability::NonNullable);
        assert_eq!(u64_scalar.nbytes(), 8);

        let f32_scalar = Scalar::primitive(3.5f32, Nullability::NonNullable);
        assert_eq!(f32_scalar.nbytes(), 4);

        let f64_scalar = Scalar::primitive(3.5f64, Nullability::NonNullable);
        assert_eq!(f64_scalar.nbytes(), 8);

        // Test UTF-8 scalar
        let utf8_scalar = Scalar::utf8("hello", Nullability::NonNullable);
        assert_eq!(utf8_scalar.nbytes(), 5);

        let empty_utf8 = Scalar::utf8("", Nullability::NonNullable);
        assert_eq!(empty_utf8.nbytes(), 0);

        // Test binary scalar
        let binary_scalar = Scalar::binary(
            ByteBuffer::from(vec![1u8, 2, 3, 4]),
            Nullability::NonNullable,
        );
        assert_eq!(binary_scalar.nbytes(), 4);

        // Test struct scalar
        let struct_scalar = Scalar::struct_(
            DType::struct_(
                [
                    ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                    ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ],
                Nullability::NonNullable,
            ),
            vec![
                Scalar::primitive(42i32, Nullability::NonNullable),
                Scalar::primitive(100i64, Nullability::NonNullable),
            ],
        );
        assert_eq!(struct_scalar.nbytes(), 4 + 8); // i32 + i64

        // Test list scalar
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );
        assert_eq!(list_scalar.nbytes(), 3 * 4); // 3 * i32

        // Test extension scalar
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));
        let ext_scalar = Scalar::extension(
            ext_dtype,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        assert_eq!(ext_scalar.nbytes(), 4); // i32 storage
    }

    #[test]
    fn test_decimal_nbytes() {
        use vortex_dtype::{DECIMAL128_MAX_PRECISION, DecimalDType};

        use crate::decimal::DecimalValue;

        // Test decimal with precision <= 38 (should use i128 = 16 bytes)
        let decimal_low_precision = Scalar::decimal(
            DecimalValue::I128(123456789),
            DecimalDType::new(DECIMAL128_MAX_PRECISION, 2), // precision 38
            Nullability::NonNullable,
        );
        assert_eq!(
            decimal_low_precision.nbytes(),
            16,
            "Decimals with precision <= 38 should be 16 bytes (i128)"
        );

        // Test decimal with precision > 38 (should use i256 = 32 bytes)
        let decimal_high_precision = Scalar::decimal(
            DecimalValue::I128(123456789),
            DecimalDType::new(DECIMAL128_MAX_PRECISION + 1, 2), // precision 39
            Nullability::NonNullable,
        );
        assert_eq!(
            decimal_high_precision.nbytes(),
            32,
            "Decimals with precision > 38 should be 32 bytes (i256)"
        );

        // Test various precision boundaries
        let decimal_p10 = Scalar::decimal(
            DecimalValue::I32(12345),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        assert_eq!(
            decimal_p10.nbytes(),
            16,
            "Decimal with precision 10 should be 16 bytes"
        );

        let decimal_p38 = Scalar::decimal(
            DecimalValue::I64(123456789),
            DecimalDType::new(38, 4),
            Nullability::NonNullable,
        );
        assert_eq!(
            decimal_p38.nbytes(),
            16,
            "Decimal with precision 38 should be 16 bytes"
        );

        let decimal_p50 = Scalar::decimal(
            DecimalValue::I128(123456789),
            DecimalDType::new(50, 5),
            Nullability::NonNullable,
        );
        assert_eq!(
            decimal_p50.nbytes(),
            32,
            "Decimal with precision 50 should be 32 bytes"
        );

        // Test null decimal - should still report size based on precision
        let null_decimal_low = Scalar::null(DType::Decimal(
            DecimalDType::new(20, 2),
            Nullability::Nullable,
        ));
        assert_eq!(
            null_decimal_low.nbytes(),
            16,
            "Null decimal with low precision should still report 16 bytes"
        );

        let null_decimal_high = Scalar::null(DType::Decimal(
            DecimalDType::new(40, 2),
            Nullability::Nullable,
        ));
        assert_eq!(
            null_decimal_high.nbytes(),
            32,
            "Null decimal with high precision should still report 32 bytes"
        );
    }

    #[test]
    fn test_scalar_nbytes_with_nulls() {
        // Test null string
        let null_utf8 = Scalar::null(DType::Utf8(Nullability::Nullable));
        assert_eq!(null_utf8.nbytes(), 0);

        // Test null binary
        let null_binary = Scalar::null(DType::Binary(Nullability::Nullable));
        assert_eq!(null_binary.nbytes(), 0);

        // Test struct with null fields
        let struct_with_null = Scalar::struct_(
            DType::struct_(
                [
                    ("a", DType::Primitive(PType::I32, Nullability::Nullable)),
                    ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ],
                Nullability::NonNullable,
            ),
            vec![
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(100i64, Nullability::NonNullable),
            ],
        );
        // Primitive null fields still count their byte width
        assert_eq!(struct_with_null.nbytes(), 4 + 8);

        // Test list with null elements
        let list_with_null = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                Scalar::primitive(3i32, Nullability::Nullable),
            ],
            Nullability::NonNullable,
        );
        // Primitive null elements still count their byte width
        assert_eq!(list_with_null.nbytes(), 3 * 4); // 3 i32 values (including null)
    }

    #[test]
    fn test_scalar_into_nullable() {
        let non_nullable = Scalar::primitive(42i32, Nullability::NonNullable);
        assert_eq!(non_nullable.dtype().nullability(), Nullability::NonNullable);

        let nullable = non_nullable.into_nullable();
        assert_eq!(nullable.dtype().nullability(), Nullability::Nullable);
        assert_eq!(nullable.as_primitive().typed_value::<i32>(), Some(42));

        // Test with already nullable scalar
        let already_nullable = Scalar::primitive(42i32, Nullability::Nullable);
        let still_nullable = already_nullable.into_nullable();
        assert_eq!(still_nullable.dtype().nullability(), Nullability::Nullable);
    }

    #[test]
    fn test_scalar_into_parts() {
        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let (dtype, value) = scalar.into_parts();

        assert_eq!(
            dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        match value {
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(v))) => {
                assert_eq!(v, 42);
            }
            _ => panic!("Expected I32 primitive value"),
        }
    }

    #[test]
    fn test_scalar_into_value() {
        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let value = scalar.into_value();

        match value {
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(v))) => {
                assert_eq!(v, 42);
            }
            _ => panic!("Expected I32 primitive value"),
        }
    }

    #[test]
    fn test_scalar_is_valid_is_null() {
        let valid_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        assert!(valid_scalar.is_valid());
        assert!(!valid_scalar.is_null());

        let null_scalar = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        assert!(!null_scalar.is_valid());
        assert!(null_scalar.is_null());
    }

    #[test]
    fn test_scalar_as_ref() {
        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let scalar_ref: &Scalar = scalar.as_ref();
        assert_eq!(scalar_ref, &scalar);
    }

    #[test]
    fn test_scalar_from_option() {
        // Test Some value
        let some_value: Option<i32> = Some(42);
        let scalar = Scalar::from(some_value);
        assert_eq!(
            scalar.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        assert_eq!(scalar.as_primitive().typed_value::<i32>(), Some(42));

        // Test None value
        let none_value: Option<i32> = None;
        let null_scalar = Scalar::from(none_value);
        assert_eq!(
            null_scalar.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        assert!(null_scalar.is_null());
    }

    #[test]
    fn test_scalar_from_primitive_scalar() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let pscalar = crate::PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(42))),
        )
        .unwrap();

        let scalar = Scalar::from(pscalar);
        assert_eq!(scalar.dtype(), &dtype);
        assert_eq!(scalar.as_primitive().typed_value::<i32>(), Some(42));
    }

    #[test]
    fn test_scalar_from_decimal_scalar() {
        use vortex_dtype::DecimalDType;

        use crate::decimal::{DecimalScalar, DecimalValue};

        let decimal_dtype = DecimalDType::new(10, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let dscalar = DecimalScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Decimal(DecimalValue::I32(12345))),
        )
        .unwrap();

        let scalar = Scalar::from(dscalar);
        assert_eq!(scalar.dtype(), &dtype);
        assert_eq!(
            scalar.as_decimal().decimal_value(),
            Some(DecimalValue::I32(12345))
        );
    }

    #[test]
    fn test_scalar_from_vec_macros() {
        // Test Vec<u16>
        let vec_u16 = vec![1u16, 2, 3];
        let scalar = Scalar::from(vec_u16);
        assert!(matches!(scalar.dtype(), DType::List(..)));
        assert_eq!(scalar.as_list().len(), 3);

        // Test Vec<i32>
        let vec_i32 = vec![10i32, 20, 30];
        let scalar = Scalar::from(vec_i32);
        assert!(matches!(scalar.dtype(), DType::List(..)));
        assert_eq!(scalar.as_list().len(), 3);

        // Test Vec<f64>
        let vec_f64 = vec![1.1f64, 2.2, 3.3];
        let scalar = Scalar::from(vec_f64);
        assert!(matches!(scalar.dtype(), DType::List(..)));
        assert_eq!(scalar.as_list().len(), 3);

        // Test Vec<String>
        let vec_string = vec!["hello".to_string(), "world".to_string()];
        let scalar = Scalar::from(vec_string);
        assert!(matches!(scalar.dtype(), DType::List(..)));
        assert_eq!(scalar.as_list().len(), 2);
    }

    #[test]
    fn test_scalar_hash() {
        let mut set = HashSet::new();

        // Add various scalar types
        set.insert(Scalar::null(DType::Null));
        set.insert(Scalar::bool(true, Nullability::NonNullable));
        set.insert(Scalar::primitive(42i32, Nullability::NonNullable));
        set.insert(Scalar::utf8("test", Nullability::NonNullable));

        // Test that duplicates are not added
        assert_eq!(set.len(), 4);
        set.insert(Scalar::primitive(42i32, Nullability::NonNullable));
        assert_eq!(set.len(), 4); // Should still be 4

        // Test that different values hash differently
        set.insert(Scalar::primitive(43i32, Nullability::NonNullable));
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_scalar_partial_ord_incompatible_types() {
        let int_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);

        // Different types should return None for partial_cmp
        assert_eq!(int_scalar.partial_cmp(&bool_scalar), None);
        assert_eq!(bool_scalar.partial_cmp(&int_scalar), None);
    }

    #[test]
    fn test_scalar_partial_ord_same_type() {
        let scalar1 = Scalar::primitive(10i32, Nullability::NonNullable);
        let scalar2 = Scalar::primitive(20i32, Nullability::NonNullable);
        let scalar3 = Scalar::primitive(10i32, Nullability::NonNullable);

        assert_eq!(
            scalar1.partial_cmp(&scalar2),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            scalar2.partial_cmp(&scalar1),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            scalar1.partial_cmp(&scalar3),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_scalar_eq() {
        let scalar1 = Scalar::primitive(42i32, Nullability::NonNullable);
        let scalar2 = Scalar::primitive(42i32, Nullability::NonNullable);
        let scalar3 = Scalar::primitive(43i32, Nullability::NonNullable);

        assert_eq!(scalar1, scalar2);
        assert_ne!(scalar1, scalar3);
    }
}
