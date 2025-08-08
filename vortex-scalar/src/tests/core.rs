// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core tests for scalar functionality including casting, coercion, and default values.

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, ExtDType, ExtID, FieldDType, Nullability, PType, StructFields};
    use vortex_error::VortexExpect;

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

    #[rstest]
    fn null_can_cast_to_anything_nullable(
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("a"),
                Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
                None,
            ))),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("b"),
                Arc::from(DType::Utf8(Nullability::Nullable)),
                None,
            )))
        )]
        source_dtype: DType,
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("a"),
                Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
                None,
            ))),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("b"),
                Arc::from(DType::Utf8(Nullability::Nullable)),
                None,
            )))
        )]
        target_dtype: DType,
    ) {
        assert_eq!(
            Scalar::null(source_dtype)
                .cast(&target_dtype)
                .unwrap()
                .dtype(),
            &target_dtype
        );
    }

    #[test]
    fn list_casts() {
        let list = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(Arc::from([ScalarValue(
                InnerScalarValue::Primitive(PValue::U16(6)),
            )]))),
        );

        let target_u32 = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list.cast(&target_u32).unwrap().dtype(), &target_u32);

        let target_u32_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert_eq!(
            list.cast(&target_u32_nonnull).unwrap().dtype(),
            &target_u32_nonnull
        );

        let target_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::NonNullable,
        );
        assert_eq!(list.cast(&target_nonnull).unwrap().dtype(), &target_nonnull);

        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list.cast(&target_u8).unwrap().dtype(), &target_u8);

        let list_with_null = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(Arc::from([
                ScalarValue(InnerScalarValue::Primitive(PValue::U16(6))),
                ScalarValue(InnerScalarValue::Null),
            ]))),
        );
        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list_with_null.cast(&target_u8).unwrap().dtype(), &target_u8);

        let target_u32_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert!(list_with_null.cast(&target_u32_nonnull).is_err());
    }

    #[test]
    fn cast_to_from_extension_types() {
        let apples = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            None,
        );
        let ext_dtype = DType::Extension(Arc::from(apples.clone()));
        let ext_scalar = Scalar::new(ext_dtype.clone(), ScalarValue(InnerScalarValue::Bool(true)));
        let storage_scalar = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(1000))),
        );

        // to self
        let expected_dtype = &ext_dtype;
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // to nullable self
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type
        let expected_dtype = apples.storage_dtype();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type, nullable
        let expected_dtype = &apples.storage_dtype().as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension, nullable
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *compatible* storage type to extension
        let storage_scalar_u64 = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(1000))),
        );
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar_u64.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *incompatible* storage type to extension
        let apples_u8 = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U8, Nullability::NonNullable)),
            None,
        );
        let expected_dtype = &DType::Extension(Arc::from(apples_u8));
        let result = storage_scalar.cast(expected_dtype);
        assert!(
            result
                .as_ref()
                .is_err_and(|err| { err.to_string().contains("Cannot cast u16 to u8") }),
            "{result:?}"
        );
    }

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
    fn test_f16_coercion_from_u64() {
        let f16_value = f16::from_f32(5.722046e-6);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_f16_no_coercion_from_u32() {
        let f16_value = f16::from_f32(0.42);
        let u32_bits = f16_value.to_bits() as u32;

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_bits))),
        );

        // No coercion expected from u32
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))) => {
                assert_eq!(*v, u32_bits);
            }
            _ => panic!("Expected U32 value (no coercion)"),
        }
    }

    #[test]
    fn test_f16_no_coercion_from_u16() {
        let f16_value = f16::from_f32(1.5);
        let u16_bits = f16_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(u16_bits))),
        );

        // No coercion expected from u16
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(v))) => {
                assert_eq!(*v, u16_bits);
            }
            _ => panic!("Expected U16 value (no coercion)"),
        }
    }

    #[test]
    fn test_f32_no_coercion_from_u32() {
        let f32_value = std::f32::consts::PI;
        let u32_bits = f32_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F32, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_bits))),
        );

        // No coercion expected from u32
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))) => {
                assert_eq!(*v, u32_bits);
            }
            _ => panic!("Expected U32 value (no coercion)"),
        }
    }

    #[test]
    fn test_f64_no_coercion_from_u64() {
        let f64_value = std::f64::consts::E;
        let u64_bits = f64_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F64, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        // No coercion expected from u64
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(v))) => {
                assert_eq!(*v, u64_bits);
            }
            _ => panic!("Expected U64 value (no coercion)"),
        }
    }

    #[test]
    fn test_struct_field_coercion() {
        let f16_value = f16::from_f32(0.42);
        let f32_value = std::f32::consts::PI;

        let struct_dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
                ),
                (
                    "b",
                    FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable)),
                ),
                (
                    "c",
                    FieldDType::from(DType::Primitive(PType::F32, Nullability::NonNullable)),
                ),
            ]),
            Nullability::NonNullable,
        );

        let field_values = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(42))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64,
            ))),
            ScalarValue(InnerScalarValue::Primitive(PValue::F32(f32_value))),
        ];

        let scalar = Scalar::new(
            struct_dtype,
            ScalarValue(InnerScalarValue::List(field_values.into())),
        );

        let struct_scalar = scalar.as_struct();
        let fields = struct_scalar.fields().unwrap();

        // Check first field (no coercion needed)
        assert_eq!(fields[0].as_primitive().pvalue().unwrap(), PValue::U32(42));

        // Check second field (f16 coerced from u64)
        assert_eq!(
            fields[1].as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );

        // Check third field (no coercion needed)
        assert_eq!(
            fields[2].as_primitive().pvalue().unwrap(),
            PValue::F32(f32_value)
        );
    }

    #[test]
    fn test_no_coercion_for_matching_types() {
        // Test that when types already match, no coercion happens
        let i32_value = 42i32;
        let scalar = Scalar::new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(i32_value))),
        );

        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(v))) => {
                assert_eq!(*v, i32_value);
            }
            _ => panic!("Expected I32 value"),
        }
    }

    #[test]
    fn test_list_element_coercion() {
        let f16_value1 = f16::from_f32(1.0);
        let f16_value2 = f16::from_f32(2.0);

        let list_dtype = DType::List(
            Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let elements = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value1.to_bits() as u64,
            ))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value2.to_bits() as u64,
            ))),
        ];

        let scalar = Scalar::new(
            list_dtype,
            ScalarValue(InnerScalarValue::List(elements.into())),
        );

        let list_scalar = scalar.as_list();
        let elements = list_scalar.elements().unwrap();

        for (i, expected) in [f16_value1, f16_value2].iter().enumerate() {
            assert_eq!(
                elements[i].as_primitive().pvalue().unwrap(),
                PValue::F16(*expected)
            );
        }
    }

    #[test]
    fn test_coercion_with_overflow_protection() {
        // Test that values too large for target type are not coerced
        let large_u64 = u64::MAX;

        // This should NOT be coerced to F16 because it's too large
        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(large_u64))),
        );

        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(v))) => {
                assert_eq!(*v, large_u64);
            }
            _ => panic!("Expected U64 value to remain unchanged when too large for F16"),
        }
    }

    #[test]
    fn test_extension_dtype_coercion() {
        // Create an extension type with f16 storage
        let ext_id = ExtID::new("test_f16_ext".into());
        let storage_dtype = Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, storage_dtype, None));

        // Test f16 value stored as u64 gets coerced through extension type
        let f16_value = f16::from_f32(0.42);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        // Verify the value was coerced to f16
        assert_eq!(
            scalar
                .as_extension()
                .storage()
                .as_primitive()
                .pvalue()
                .unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_scalar_nbytes() {
        use vortex_buffer::ByteBuffer;

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
        use vortex_utils::aliases::hash_set::HashSet;

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

    #[test]
    fn test_extension_dtype_nested_struct_coercion() {
        // Create an extension type with struct storage that contains f16 field
        let ext_id = ExtID::new("test_struct_ext".into());
        let struct_dtype = Arc::new(DType::Struct(
            StructFields::from_iter([
                (
                    "id",
                    FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
                ),
                (
                    "value",
                    FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable)),
                ),
            ]),
            Nullability::NonNullable,
        ));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, struct_dtype, None));

        // Create struct value with f16 stored as u64
        let f16_value = f16::from_f32(1.5);
        let field_values = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(123))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64,
            ))),
        ];

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::List(field_values.into())),
        );

        // Verify the struct field was coerced
        let list_elems = scalar
            .as_extension()
            .storage()
            .as_struct()
            .fields()
            .vortex_expect("non null");
        assert_eq!(
            list_elems[0].as_primitive().pvalue().unwrap(),
            PValue::U32(123)
        );
        assert_eq!(
            list_elems[1].as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );
    }
}
