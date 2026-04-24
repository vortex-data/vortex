// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for internal consistency and round-tripping of scalar types.
//!
//! These tests ensure that conversions between different representations
//! (Scalar, ScalarValue, specialized types, protobuf) maintain consistency
//! and preserve data integrity.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_proto::scalar as pb;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
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
            Scalar::primitive(std::f32::consts::PI, Nullability::NonNullable),
            Scalar::primitive(std::f64::consts::E, Nullability::NonNullable),
        ];

        for scalar in values {
            let value = scalar.value().cloned();
            let dtype = scalar.dtype().clone();
            let reconstructed = Scalar::new(dtype, value);
            assert_eq!(scalar, reconstructed);
        }
    }

    // Test that null scalars maintain their type information
    #[test]
    fn test_null_scalar_type_preservation() {
        let null_scalars = vec![
            Scalar::null_native::<i8>(),
            Scalar::null_native::<i16>(),
            Scalar::null_native::<i32>(),
            Scalar::null_native::<i64>(),
            Scalar::null_native::<u8>(),
            Scalar::null_native::<u16>(),
            Scalar::null_native::<u32>(),
            Scalar::null_native::<u64>(),
            Scalar::null_native::<f32>(),
            Scalar::null_native::<f64>(),
            Scalar::null_native::<bool>(),
            Scalar::null_native::<String>(),
        ];

        for scalar in null_scalars {
            assert!(scalar.is_null());
            let dtype = scalar.dtype().clone();
            let value = scalar.value().cloned();
            let reconstructed = Scalar::new(dtype.clone(), value);
            assert_eq!(scalar, reconstructed);
            assert_eq!(scalar.dtype(), reconstructed.dtype());
        }
    }

    // Test conversions between Scalar and specialized scalar types
    #[test]
    fn test_specialized_scalar_conversions() {
        // Test PrimitiveScalar
        let int_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let primitive_scalar = int_scalar.as_primitive();
        assert_eq!(primitive_scalar.typed_value::<i32>().unwrap(), 42);
        let reconstructed = Scalar::from(primitive_scalar);
        assert_eq!(int_scalar, reconstructed);

        // Test BoolScalar
        let bool_scalar = Scalar::bool(true, Nullability::NonNullable);
        let bool_specialized = bool_scalar.as_bool();
        assert!(bool_specialized.value().unwrap());

        // Test Utf8Scalar
        let utf8_scalar = Scalar::utf8("hello".to_string(), Nullability::NonNullable);
        let utf8_specialized = utf8_scalar.as_utf8();
        assert_eq!(utf8_specialized.value().unwrap().as_str(), "hello");

        // Test BinaryScalar
        let binary_scalar = Scalar::binary(vec![1, 2, 3, 4], Nullability::NonNullable);
        let binary_specialized = binary_scalar.as_binary();
        assert_eq!(
            binary_specialized.value().unwrap().as_slice(),
            &[1, 2, 3, 4]
        );
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

    // Test list scalar round-trips
    #[test]
    fn test_list_scalar_round_trip() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
            Scalar::primitive(3i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children.clone(), Nullability::NonNullable);

        // Extract as ListScalar
        let list_specialized = list_scalar.as_list();
        assert_eq!(list_specialized.len(), 3);

        // Extract as Vec<i32>
        let vec: Vec<i32> = Vec::try_from(&list_scalar).unwrap();
        assert_eq!(vec, vec![1, 2, 3]);

        // Check that elements match
        for (i, expected) in children.iter().enumerate() {
            let elem = list_specialized.element(i).unwrap();
            assert_eq!(&elem, expected);
        }
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
        assert_eq!(nested_list, round_tripped_nested);

        // Test large binary data
        let large_binary = vec![42u8; 10000];
        let binary_scalar = Scalar::binary(large_binary.clone(), Nullability::NonNullable);
        let pb_binary = pb::Scalar::from(&binary_scalar);
        let round_tripped_binary = Scalar::from_proto(&pb_binary, &SESSION).unwrap();
        assert_eq!(binary_scalar, round_tripped_binary);

        // Verify the data is preserved
        let extracted: ByteBuffer = ByteBuffer::try_from(&round_tripped_binary).unwrap();
        assert_eq!(extracted.as_slice(), &large_binary);
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

    /// Verifies that [`Scalar::nbytes`] matches the length of the proto-serialized scalar value.
    #[rstest]
    #[case::null_i32(Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)))]
    #[case::bool_true(Scalar::from(true))]
    #[case::bool_false(Scalar::from(false))]
    #[case::i8(Scalar::from(i8::MAX))]
    #[case::i16(Scalar::from(i16::MAX))]
    #[case::i32(Scalar::from(i32::MAX))]
    #[case::i64(Scalar::from(i64::MAX))]
    #[case::u8(Scalar::from(u8::MAX))]
    #[case::u16(Scalar::from(u16::MAX))]
    #[case::u32(Scalar::from(u32::MAX))]
    #[case::u64(Scalar::from(u64::MAX))]
    #[case::f32(Scalar::from(f32::MAX))]
    #[case::f64(Scalar::from(f64::MAX))]
    #[case::utf8_empty(Scalar::from(""))]
    #[case::utf8_short(Scalar::from("hello"))]
    #[case::utf8_long(Scalar::from("x".repeat(2048).as_str()))]
    #[case::binary_empty(Scalar::binary(Vec::<u8>::new(), Nullability::NonNullable))]
    #[case::binary_short(Scalar::binary(vec![1u8, 2, 3], Nullability::NonNullable))]
    fn test_nbytes_approx_eq_to_proto_bytes(#[case] scalar: Scalar) {
        let proto_bytes: Vec<u8> = ScalarValue::to_proto_bytes(scalar.value());
        let diff = (scalar.approx_nbytes() as isize - proto_bytes.len() as isize).abs();

        // NOTE: THE 4 HERE IS COMPLETELY ARBITRARY!!!
        assert!(
            diff <= 4,
            "nbytes() should be within 4 of proto-serialized length for {:?}, got {} vs {}",
            scalar,
            scalar.approx_nbytes(),
            proto_bytes.len(),
        );
    }
}
