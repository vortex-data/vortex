// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for internal consistency and round-tripping of scalar types.
//!
//! These tests ensure that conversions between different representations
//! (Scalar, ScalarValue, specialized types, protobuf) maintain consistency
//! and preserve data integrity.

#[cfg(test)]
mod tests {
    use std::f32;
    use std::f64;
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::proto::scalar as pb;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::tests::SESSION;

    // Test that primitive scalars round-trip through ScalarValue
    #[test]
    fn test_primitive_scalar_to_scalar_value_round_trip() {
        let values: Vec<Scalar> = vec![
            Scalar::primitive(42i8, Nullability::NonNullable),
            Scalar::primitive(1000i16, Nullability::NonNullable),
            Scalar::primitive(100000i32, Nullability::NonNullable),
            Scalar::primitive(10000000000i64, Nullability::NonNullable),
            Scalar::primitive(200u8, Nullability::NonNullable),
            Scalar::primitive(50000u16, Nullability::NonNullable),
            Scalar::primitive(4000000000u32, Nullability::NonNullable),
            Scalar::primitive(18446744073709551615u64, Nullability::NonNullable),
            Scalar::primitive(f32::consts::PI, Nullability::NonNullable),
            Scalar::primitive(f64::consts::E, Nullability::NonNullable),
        ];

        for scalar in values {
            let value = scalar.value().cloned();
            let dtype = scalar.dtype().clone();
            let reconstructed = Scalar::new(dtype, value);
            assert_eq!(scalar.dtype(), reconstructed.dtype());
            assert_eq!(scalar, reconstructed);
        }
    }

    #[test]
    fn test_null_scalar_type_preservation() {
        let cases = [
            (
                Scalar::null_native::<i8>(),
                DType::Primitive(PType::I8, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<i16>(),
                DType::Primitive(PType::I16, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<i32>(),
                DType::Primitive(PType::I32, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<i64>(),
                DType::Primitive(PType::I64, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<u8>(),
                DType::Primitive(PType::U8, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<u16>(),
                DType::Primitive(PType::U16, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<u32>(),
                DType::Primitive(PType::U32, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<u64>(),
                DType::Primitive(PType::U64, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<f32>(),
                DType::Primitive(PType::F32, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<f64>(),
                DType::Primitive(PType::F64, Nullability::Nullable),
            ),
            (
                Scalar::null_native::<bool>(),
                DType::Bool(Nullability::Nullable),
            ),
            (
                Scalar::null_native::<String>(),
                DType::Utf8(Nullability::Nullable),
            ),
        ];

        for (scalar, expected_dtype) in cases {
            assert!(scalar.is_null());
            assert_eq!(scalar.dtype(), &expected_dtype);
            let reconstructed = Scalar::new(expected_dtype.clone(), scalar.value().cloned());
            assert!(reconstructed.is_null());
            assert_eq!(reconstructed.dtype(), &expected_dtype);
        }
    }

    // Test that From<T> and TryFrom<&Scalar> for T are consistent
    #[test]
    fn test_from_try_from_consistency() {
        // Test with various primitive types
        let value_i32 = 42i32;
        let scalar_i32 = Scalar::from(value_i32);
        let extracted_i32: i32 = i32::try_from(&scalar_i32).unwrap();
        assert_eq!(value_i32, extracted_i32);

        let value_u64 = 1000000u64;
        let scalar_u64 = Scalar::from(value_u64);
        let extracted_u64: u64 = u64::try_from(&scalar_u64).unwrap();
        assert_eq!(value_u64, extracted_u64);

        let value_bool = true;
        let scalar_bool = Scalar::from(value_bool);
        let extracted_bool: bool = bool::try_from(&scalar_bool).unwrap();
        assert_eq!(value_bool, extracted_bool);

        let value_str = "test string";
        let scalar_str = Scalar::from(value_str);
        let extracted_str: String = String::try_from(&scalar_str).unwrap();
        assert_eq!(value_str, extracted_str);
    }

    // Test Option<T> conversions
    #[test]
    fn test_option_conversions() {
        // Test Some values
        let some_value = Some(42i32);
        let scalar_some = Scalar::from(some_value);
        let extracted_some: Option<i32> = Option::try_from(&scalar_some).unwrap();
        assert_eq!(some_value, extracted_some);

        // Test None values
        let none_value: Option<i32> = None;
        let scalar_none = Scalar::from(none_value);
        let extracted_none: Option<i32> = Option::try_from(&scalar_none).unwrap();
        assert_eq!(none_value, extracted_none);
    }

    // Test decimal scalar round-trips
    #[test]
    fn test_decimal_scalar_round_trip() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Test various decimal value types.
        let decimal_values = vec![
            DecimalValue::I8(100),
            DecimalValue::I16(10000),
            DecimalValue::I32(1000000),
            DecimalValue::I64(10000000),
            DecimalValue::I128(100000000),
            DecimalValue::I256(i256::from_i128(1000000000)),
        ];

        for value in decimal_values {
            let scalar = Scalar::decimal(value, decimal_dtype, Nullability::NonNullable);
            let decimal_specialized = scalar.as_decimal();

            match decimal_specialized.decimal_value() {
                Some(extracted) => assert_eq!(extracted, value),
                None => panic!("Expected decimal value, got None"),
            }

            // Test round-trip through ScalarValue
            let scalar_value = scalar.value().cloned();
            let dtype = scalar.dtype().clone();
            let reconstructed = Scalar::new(dtype, scalar_value);
            assert_eq!(scalar.dtype(), reconstructed.dtype());
            assert_eq!(scalar, reconstructed);
        }
    }

    // Test protobuf round-trips with edge cases
    #[test]
    fn test_protobuf_edge_cases() {
        // Test empty list
        let empty_list = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        );
        let pb_empty = pb::Scalar::from(&empty_list);
        let round_tripped = Scalar::from_proto(&pb_empty, &SESSION).unwrap();
        assert_eq!(empty_list.dtype(), round_tripped.dtype());
        assert_eq!(empty_list, round_tripped);

        // Test nested lists
        let inner_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let outer_dtype = Arc::new(DType::List(
            Arc::clone(&inner_dtype),
            Nullability::NonNullable,
        ));

        let inner_list1 = Scalar::list(
            inner_dtype,
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let nested_list = Scalar::list(outer_dtype, vec![inner_list1], Nullability::NonNullable);

        let pb_nested = pb::Scalar::from(&nested_list);
        let round_tripped_nested = Scalar::from_proto(&pb_nested, &SESSION).unwrap();
        assert_eq!(nested_list.dtype(), round_tripped_nested.dtype());
        assert_eq!(nested_list, round_tripped_nested);

        // Test large binary data
        let large_binary = vec![42u8; 10000];
        let binary_scalar = Scalar::binary(large_binary.clone(), Nullability::NonNullable);
        let pb_binary = pb::Scalar::from(&binary_scalar);
        let round_tripped_binary = Scalar::from_proto(&pb_binary, &SESSION).unwrap();
        assert_eq!(binary_scalar.dtype(), round_tripped_binary.dtype());
        assert_eq!(binary_scalar, round_tripped_binary);

        // Verify the data is preserved
        let extracted: ByteBuffer = ByteBuffer::try_from(&round_tripped_binary).unwrap();
        assert_eq!(extracted.as_slice(), &large_binary);
    }

    // Test that fixed-size list scalars round-trip through protobuf. This is
    // the path taken by a ConstantArray of a fixed-size list dtype (e.g. a
    // UUID column stored as fixed_size_list(u8)[16]), whose scalar metadata
    // is serialized as a protobuf ListValue.
    #[test]
    fn test_protobuf_fixed_size_list_round_trip() {
        let fsl = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            (0u8..16)
                .map(|i| Scalar::primitive(i, Nullability::NonNullable))
                .collect(),
            Nullability::NonNullable,
        );

        let pb_fsl = pb::Scalar::from(&fsl);
        let round_tripped = Scalar::from_proto(&pb_fsl, &SESSION).unwrap();
        assert_eq!(fsl.dtype(), round_tripped.dtype());
        assert_eq!(fsl, round_tripped);
    }

    // Test that nullable and non-nullable types are preserved
    #[test]
    fn test_nullability_preservation() {
        let nullable_scalar = Scalar::primitive(42i32, Nullability::Nullable);
        let non_nullable_scalar = Scalar::primitive(42i32, Nullability::NonNullable);

        assert_ne!(nullable_scalar.dtype(), non_nullable_scalar.dtype());

        // Test through protobuf
        let pb_nullable = pb::Scalar::from(&nullable_scalar);
        let pb_non_nullable = pb::Scalar::from(&non_nullable_scalar);

        let recovered_nullable = Scalar::from_proto(&pb_nullable, &SESSION).unwrap();
        let recovered_non_nullable = Scalar::from_proto(&pb_non_nullable, &SESSION).unwrap();

        assert_eq!(nullable_scalar.dtype(), recovered_nullable.dtype());
        assert_eq!(non_nullable_scalar.dtype(), recovered_non_nullable.dtype());
        assert_ne!(recovered_nullable.dtype(), recovered_non_nullable.dtype());
        assert_eq!(
            recovered_nullable.as_primitive().typed_value::<i32>(),
            Some(42)
        );
        assert_eq!(
            recovered_non_nullable.as_primitive().typed_value::<i32>(),
            Some(42)
        );
    }

    // Test usize conversions (which may be architecture-dependent)
    #[test]
    fn test_usize_conversions() {
        let value_usize = 12345usize;
        let scalar_usize = Scalar::from(value_usize);
        let extracted_usize: usize = usize::try_from(&scalar_usize).unwrap();
        assert_eq!(value_usize, extracted_usize);
    }

    // Test error cases for conversions
    #[test]
    fn test_conversion_errors() {
        // Try to convert a string scalar to an integer
        let string_scalar = Scalar::utf8("not a number".to_string(), Nullability::NonNullable);
        let result: Result<i32, _> = i32::try_from(&string_scalar);
        assert!(result.is_err());

        // Try to convert an integer scalar to a list.
        let int_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        assert!(int_scalar.as_list_opt().is_none());

        // Try to convert a boolean to a decimal.
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);
        assert!(bool_scalar.as_decimal_opt().is_none());
    }
}
