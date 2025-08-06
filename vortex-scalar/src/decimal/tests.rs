// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for decimal scalar casting functionality.

#![allow(clippy::disallowed_types, clippy::panic)]

use rstest::rstest;
use vortex_dtype::{DType, DecimalDType, Nullability, PType};
use vortex_utils::aliases::hash_set::HashSet;

use crate::{DecimalValue, Scalar, i256};

#[rstest]
#[case(DecimalValue::I8(100), DecimalValue::I8(100))]
#[case(DecimalValue::I16(0), DecimalValue::I256(i256::ZERO))]
#[case(DecimalValue::I8(100), DecimalValue::I128(100))]
fn test_decimal_value_eq(#[case] left: DecimalValue, #[case] right: DecimalValue) {
    assert_eq!(left, right);
}

#[rstest]
#[case(DecimalValue::I128(10), DecimalValue::I8(11))]
#[case(DecimalValue::I256(i256::ZERO), DecimalValue::I16(10))]
#[case(DecimalValue::I128(-1_000), DecimalValue::I8(1))]
fn test_decimal_value_cmp(#[case] lower: DecimalValue, #[case] upper: DecimalValue) {
    assert!(lower < upper, "expected {lower} < {upper}");
}

#[test]
fn test_hash() {
    let mut set = HashSet::new();
    set.insert(DecimalValue::I8(100));
    set.insert(DecimalValue::I16(100));
    set.insert(DecimalValue::I32(100));
    set.insert(DecimalValue::I64(100));
    set.insert(DecimalValue::I128(100));
    set.insert(DecimalValue::I256(i256::from_i128(100)));
    assert_eq!(set.len(), 1);
}

#[test]
fn test_decimal_cast_to_primitive() {
    // Create a decimal with value 123.45 (scale=2, so stored as 12345)
    let decimal_scalar = Scalar::decimal(
        DecimalValue::I32(12345),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );

    // Cast to f64 should give us 123.45
    let float_result = decimal_scalar
        .cast(&DType::Primitive(PType::F64, Nullability::NonNullable))
        .unwrap();
    let float_value: f64 = float_result.try_into().unwrap();
    assert!((float_value - 123.45).abs() < 0.001);

    // Cast to i32 should give us 123 (truncated)
    let int_result = decimal_scalar
        .cast(&DType::Primitive(PType::I32, Nullability::NonNullable))
        .unwrap();
    let int_value: i32 = int_result.try_into().unwrap();
    assert_eq!(int_value, 123);
}

#[test]
fn test_decimal_cast_null_handling() {
    // Null decimal
    let null_decimal = Scalar::null(DType::Decimal(
        DecimalDType::new(10, 2),
        Nullability::Nullable,
    ));

    // Cast null decimal to primitive should preserve null
    let result = null_decimal
        .cast(&DType::Primitive(PType::I32, Nullability::Nullable))
        .unwrap();
    assert!(result.is_null());

    // Cast null decimal to another decimal type should preserve null
    let result = null_decimal
        .cast(&DType::Decimal(
            DecimalDType::new(20, 4),
            Nullability::Nullable,
        ))
        .unwrap();
    assert!(result.is_null());
}

#[test]
fn test_decimal_cast_overflow() {
    // Large decimal value that won't fit in i8
    let decimal_scalar = Scalar::decimal(
        DecimalValue::I32(100000),
        DecimalDType::new(10, 0),
        Nullability::NonNullable,
    );

    // Cast to i8 should fail due to overflow
    let result = decimal_scalar.cast(&DType::Primitive(PType::I8, Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_decimal_cast_between_decimal_types() {
    // Decimal with different precision/scale
    let decimal_scalar = Scalar::decimal(
        DecimalValue::I32(12345),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );

    // Cast to different decimal type (currently just preserves value)
    let result = decimal_scalar
        .cast(&DType::Decimal(
            DecimalDType::new(20, 4),
            Nullability::NonNullable,
        ))
        .unwrap();

    // Value should be preserved (TODO: proper scaling logic)
    let decimal_value: Option<DecimalValue> = result.try_into().unwrap();
    assert_eq!(decimal_value, Some(DecimalValue::I32(12345)));
}

#[test]
fn test_decimal_cast_negative_values() {
    // Negative decimal value
    let decimal_scalar = Scalar::decimal(
        DecimalValue::I32(-5678),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );

    // Cast to f64 should give us -56.78
    let float_result = decimal_scalar
        .cast(&DType::Primitive(PType::F64, Nullability::NonNullable))
        .unwrap();
    let float_value: f64 = float_result.try_into().unwrap();
    assert!((float_value - (-56.78)).abs() < 0.001);

    // Cast to unsigned should fail
    let result = decimal_scalar.cast(&DType::Primitive(PType::U32, Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_decimal_cast_edge_values() {
    // Test with extreme values for each decimal type
    let test_cases = vec![
        (DecimalValue::I8(i8::MAX), DecimalDType::new(3, 0)),
        (DecimalValue::I8(i8::MIN), DecimalDType::new(3, 0)),
        (DecimalValue::I16(i16::MAX), DecimalDType::new(5, 0)),
        (DecimalValue::I16(i16::MIN), DecimalDType::new(5, 0)),
        (DecimalValue::I32(i32::MAX), DecimalDType::new(10, 0)),
        (DecimalValue::I32(i32::MIN), DecimalDType::new(10, 0)),
    ];

    for (value, dtype) in test_cases {
        let decimal_scalar = Scalar::decimal(value, dtype, Nullability::NonNullable);

        // Cast to f64 should always work for these ranges
        let result = decimal_scalar.cast(&DType::Primitive(PType::F64, Nullability::NonNullable));
        assert!(result.is_ok());
    }
}

#[test]
fn test_decimal_cast_with_scale() {
    // Test various scale factors
    let test_cases = vec![
        (1234, 0, 1234.0), // No scale
        (1234, 1, 123.4),  // Scale 1
        (1234, 2, 12.34),  // Scale 2
        (1234, 3, 1.234),  // Scale 3
        (1234, 4, 0.1234), // Scale 4
    ];

    for (value, scale, expected) in test_cases {
        let decimal_scalar = Scalar::decimal(
            DecimalValue::I32(value),
            DecimalDType::new(10, scale),
            Nullability::NonNullable,
        );

        let float_result = decimal_scalar
            .cast(&DType::Primitive(PType::F64, Nullability::NonNullable))
            .unwrap();
        let float_value: f64 = float_result.try_into().unwrap();
        assert!(
            (float_value - expected).abs() < 0.0001,
            "Expected {expected} but got {float_value} for value={value} scale={scale}"
        );
    }
}

#[test]
fn test_decimal_cast_unsupported_types() {
    let decimal_scalar = Scalar::decimal(
        DecimalValue::I32(1234),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );

    // Cast to unsupported types should fail
    let result = decimal_scalar.cast(&DType::Bool(Nullability::NonNullable));
    assert!(result.is_err());

    let result = decimal_scalar.cast(&DType::Utf8(Nullability::NonNullable));
    assert!(result.is_err());

    let result = decimal_scalar.cast(&DType::Binary(Nullability::NonNullable));
    assert!(result.is_err());
}

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
    assert_eq!(
        f64_scalar.as_primitive().typed_value::<f64>().unwrap(),
        123.45
    );
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
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("out of range for u8")
    );
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
        Some(DecimalValue::I64(123456))
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
        Some(DecimalValue::I32(10000))
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
    assert_eq!(
        i32_scalar.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
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
    assert!(
        (f64_scalar.as_primitive().typed_value::<f64>().unwrap() - 1234.567890).abs() < 0.000001
    );
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
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("out of range for u32")
    );
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
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Cannot cast decimal to")
    );
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
        assert!(result.is_ok(), "Failed to cast to {ptype:?}");
        let scalar = result.unwrap();

        // Check the value matches expected
        match ptype {
            PType::U8 => assert_eq!(
                scalar.as_primitive().typed_value::<u8>().unwrap() as u64,
                expected
            ),
            PType::U16 => assert_eq!(
                scalar.as_primitive().typed_value::<u16>().unwrap() as u64,
                expected
            ),
            PType::U32 => assert_eq!(
                scalar.as_primitive().typed_value::<u32>().unwrap() as u64,
                expected
            ),
            PType::U64 => assert_eq!(
                scalar.as_primitive().typed_value::<u64>().unwrap(),
                expected
            ),
            PType::I8 => assert_eq!(
                scalar.as_primitive().typed_value::<i8>().unwrap() as u64,
                expected
            ),
            PType::I16 => assert_eq!(
                scalar.as_primitive().typed_value::<i16>().unwrap() as u64,
                expected
            ),
            PType::I32 => assert_eq!(
                scalar.as_primitive().typed_value::<i32>().unwrap() as u64,
                expected
            ),
            PType::I64 => assert_eq!(
                scalar.as_primitive().typed_value::<i64>().unwrap() as u64,
                expected
            ),
            _ => panic!("Unexpected type"),
        }
    }
}
