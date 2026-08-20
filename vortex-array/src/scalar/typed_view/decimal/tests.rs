// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for decimal scalar casting functionality.

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::MAX_PRECISION;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::decimal_numeric_result_dtype;

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
    let float_result = &decimal_scalar
        .cast(&DType::Primitive(PType::F64, Nullability::NonNullable))
        .unwrap();
    let float_value: f64 = float_result.try_into().unwrap();
    assert!((float_value - 123.45).abs() < 0.001);

    // Cast to i32 should give us 123 (truncated)
    let int_result = &decimal_scalar
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

    // Cast to different decimal type.
    let result = decimal_scalar
        .cast(&DType::Decimal(
            DecimalDType::new(20, 4),
            Nullability::NonNullable,
        ))
        .unwrap();

    // 123.45 with scale 2 is represented as 123.4500 with scale 4.
    let decimal_value: Option<DecimalValue> = result.try_into().unwrap();
    assert_eq!(decimal_value, Some(DecimalValue::I128(1234500)));
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
    let float_result = &decimal_scalar
        .cast(&DType::Primitive(PType::F64, Nullability::NonNullable))
        .unwrap();
    let float_value: f64 = float_result.try_into().unwrap();
    assert!((float_value - (-56.78)).abs() < 0.001);

    // Cast to unsigned should fail
    let result = decimal_scalar.cast(&DType::Primitive(PType::U32, Nullability::NonNullable));
    assert!(result.is_err());
}

#[rstest]
#[case(DecimalValue::I8(i8::MAX), DecimalDType::new(3, 0))]
#[case(DecimalValue::I8(i8::MIN), DecimalDType::new(3, 0))]
#[case(DecimalValue::I16(i16::MAX), DecimalDType::new(5, 0))]
#[case(DecimalValue::I16(i16::MIN), DecimalDType::new(5, 0))]
#[case(DecimalValue::I32(i32::MAX), DecimalDType::new(10, 0))]
#[case(DecimalValue::I32(i32::MIN), DecimalDType::new(10, 0))]
fn test_decimal_cast_edge_values(#[case] value: DecimalValue, #[case] dtype: DecimalDType) {
    let decimal_scalar = Scalar::decimal(value, dtype, Nullability::NonNullable);

    // Cast to f64 should always work for these ranges
    let result = decimal_scalar.cast(&DType::Primitive(PType::F64, Nullability::NonNullable));
    assert!(result.is_ok());
}

#[rstest]
#[case(1234, 0, 1234.0)] // No scale
#[case(1234, 1, 123.4)] // Scale 1
#[case(1234, 2, 12.34)] // Scale 2
#[case(1234, 3, 1.234)] // Scale 3
#[case(1234, 4, 0.1234)] // Scale 4
fn test_decimal_cast_with_scale(#[case] value: i32, #[case] scale: i8, #[case] expected: f64) {
    let decimal_scalar = Scalar::decimal(
        DecimalValue::I32(value),
        DecimalDType::new(10, scale),
        Nullability::NonNullable,
    );

    let float_result = &decimal_scalar
        .cast(&DType::Primitive(PType::F64, Nullability::NonNullable))
        .unwrap();
    let float_value: f64 = float_result.try_into().unwrap();
    assert!(
        (float_value - expected).abs() < 0.0001,
        "Expected {expected} but got {float_value} for value={value} scale={scale}"
    );
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
    let target_dtype = DType::Decimal(DecimalDType::new(10, 4), Nullability::NonNullable);
    let result = decimal.cast(&target_dtype);
    assert!(result.is_ok());

    let casted = result.unwrap();
    assert_eq!(
        casted.as_decimal().decimal_value(),
        Some(DecimalValue::I64(1000000))
    );
}

#[test]
fn test_decimal_to_decimal_lower_scale_exact() {
    let decimal = Scalar::decimal(
        DecimalValue::I64(1234500), // 123.4500
        DecimalDType::new(10, 4),
        Nullability::NonNullable,
    );

    let casted = decimal
        .cast(&DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        ))
        .unwrap();

    assert_eq!(
        casted.as_decimal().decimal_value(),
        Some(DecimalValue::I64(12345))
    );
}

#[test]
fn test_decimal_to_decimal_lower_scale_lossy_fails() {
    let decimal = Scalar::decimal(
        DecimalValue::I64(1234567), // 123.4567
        DecimalDType::new(10, 4),
        Nullability::NonNullable,
    );

    let result = decimal.cast(&DType::Decimal(
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    ));

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("would lose precision")
    );
}

#[test]
fn test_primitive_i64_to_decimal_rescales() {
    let scalar = Scalar::primitive(42i64, Nullability::NonNullable);

    let casted = scalar
        .cast(&DType::Decimal(
            DecimalDType::new(21, 2),
            Nullability::NonNullable,
        ))
        .unwrap();

    assert_eq!(
        casted.as_decimal().decimal_value(),
        Some(DecimalValue::I128(4200))
    );
}

#[test]
fn test_primitive_to_decimal_precision_checked() {
    let scalar = Scalar::primitive(1000i32, Nullability::NonNullable);

    let result = scalar.cast(&DType::Decimal(
        DecimalDType::new(2, 0),
        Nullability::NonNullable,
    ));

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not fit in precision")
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
    use crate::dtype::i256;
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

#[rstest]
#[case(PType::U8, 5u64)]
#[case(PType::U16, 5)]
#[case(PType::U32, 5)]
#[case(PType::U64, 5)]
#[case(PType::I8, 5)]
#[case(PType::I16, 5)]
#[case(PType::I32, 5)]
#[case(PType::I64, 5)]
fn test_decimal_i8_all_primitive_casts(#[case] ptype: PType, #[case] expected: u64) {
    // Test casting from smallest decimal type to all primitive types
    let decimal = Scalar::decimal(
        DecimalValue::I8(50), // Represents 5.0 with scale=1
        DecimalDType::new(3, 1),
        Nullability::NonNullable,
    );

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
        PType::F16 | PType::F32 | PType::F64 => panic!("Unexpected type {ptype}"),
    }
}

#[test]
fn test_decimal_cast_f16() {
    use crate::dtype::half::f16;

    // Create a decimal with value 12.5 (scale=1, so stored as 125)
    let decimal = Scalar::decimal(
        DecimalValue::I16(125),
        DecimalDType::new(4, 1),
        Nullability::NonNullable,
    );

    // Cast to f16
    let result = decimal.cast(&DType::Primitive(PType::F16, Nullability::NonNullable));
    assert!(result.is_ok());
    let f16_scalar = result.unwrap();
    let f16_value: f16 = f16_scalar.as_primitive().typed_value::<f16>().unwrap();
    assert!((f16_value.to_f64() - 12.5).abs() < 0.01);
}

#[test]
fn test_decimal_cast_boundary_values() {
    // Test with U16 boundary
    let decimal = Scalar::decimal(
        DecimalValue::I32(6_553_500), // 65535.00 with scale=2
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );

    // Should succeed for U16
    let result = decimal.cast(&DType::Primitive(PType::U16, Nullability::NonNullable));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().as_primitive().typed_value::<u16>().unwrap(),
        65535
    );

    // Should fail for U16 with value 65536
    let decimal = Scalar::decimal(
        DecimalValue::I32(6_553_600), // 65536.00 with scale=2
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let result = decimal.cast(&DType::Primitive(PType::U16, Nullability::NonNullable));
    assert!(result.is_err());

    // Test with I16 boundaries
    let decimal = Scalar::decimal(
        DecimalValue::I32(3_276_700), // 32767.00 with scale=2
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let result = decimal.cast(&DType::Primitive(PType::I16, Nullability::NonNullable));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().as_primitive().typed_value::<i16>().unwrap(),
        32767
    );

    let decimal = Scalar::decimal(
        DecimalValue::I32(-3_276_800), // -32768.00 with scale=2
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let result = decimal.cast(&DType::Primitive(PType::I16, Nullability::NonNullable));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().as_primitive().typed_value::<i16>().unwrap(),
        -32768
    );

    // Should fail for I16 with value 32768
    let decimal = Scalar::decimal(
        DecimalValue::I32(3_276_800), // 32768.00 with scale=2
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let result = decimal.cast(&DType::Primitive(PType::I16, Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_decimal_partial_ord() {
    let decimal1 = Scalar::decimal(
        DecimalValue::I32(100),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let scalar1 = decimal1.as_decimal();

    let decimal2 = Scalar::decimal(
        DecimalValue::I32(200),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let scalar2 = decimal2.as_decimal();

    // Same type comparison should work
    assert!(scalar1 < scalar2);
    assert!(scalar2 > scalar1);
    assert_eq!(
        scalar1.partial_cmp(&scalar1),
        Some(std::cmp::Ordering::Equal)
    );

    // Different type comparison should return None
    let decimal3 = Scalar::decimal(
        DecimalValue::I32(100),
        DecimalDType::new(20, 4), // Different precision/scale
        Nullability::NonNullable,
    );
    let scalar3 = decimal3.as_decimal();
    assert_eq!(scalar1.partial_cmp(&scalar3), None);
}

#[test]
fn test_decimal_eq() {
    let decimal1 = Scalar::decimal(
        DecimalValue::I32(100),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let scalar1 = decimal1.as_decimal();

    let decimal2 = Scalar::decimal(
        DecimalValue::I32(100),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let scalar2 = decimal2.as_decimal();

    assert_eq!(scalar1, scalar2);

    // Different values
    let decimal3 = Scalar::decimal(
        DecimalValue::I32(200),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let scalar3 = decimal3.as_decimal();
    assert_ne!(scalar1, scalar3);
}

#[test]
fn test_decimal_value_from_unsigned() {
    // Test From implementations for unsigned types
    let v1: DecimalValue = 255u8.into();
    assert_eq!(v1, DecimalValue::I16(255));

    let v2: DecimalValue = 65535u16.into();
    assert_eq!(v2, DecimalValue::I32(65535));

    let v3: DecimalValue = 4294967295u32.into();
    assert_eq!(v3, DecimalValue::I64(4294967295));

    let v4: DecimalValue = 18446744073709551615u64.into();
    assert_eq!(v4, DecimalValue::I128(18446744073709551615));
}

#[test]
fn test_decimal_scalar_try_from_errors() {
    // Test error cases for TryFrom<DecimalScalar> for primitive types
    let decimal = Scalar::decimal(
        DecimalValue::I16(1234),
        DecimalDType::new(5, 2),
        Nullability::NonNullable,
    );
    let scalar = decimal.as_decimal();

    // Try to extract as wrong type
    let result: Result<i8, _> = scalar.try_into();
    assert!(result.is_err());

    // Try to extract from null
    let null_decimal = Scalar::null(DType::Decimal(
        DecimalDType::new(10, 2),
        Nullability::Nullable,
    ));
    let null_scalar = null_decimal.as_decimal();
    let result: Result<i32, _> = null_scalar.try_into();
    assert!(result.is_err());

    // Extract as Option from null should succeed
    let result: Result<Option<i32>, _> = null_scalar.try_into();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), None);
}

#[test]
fn test_decimal_cast_large_scale() {
    // Test with very large scale factors
    let decimal = Scalar::decimal(
        DecimalValue::I64(123456789012345), // 1234.56789012345 with scale=11
        DecimalDType::new(20, 11),
        Nullability::NonNullable,
    );

    // Cast to f64
    let result = decimal.cast(&DType::Primitive(PType::F64, Nullability::NonNullable));
    assert!(result.is_ok());
    let f64_value: f64 = (&result.unwrap()).try_into().unwrap();
    assert!((f64_value - 1234.56789012345).abs() < 0.0000000001);
}

#[test]
fn test_decimal_cast_zero_scale() {
    // Test with zero scale (integer values)
    let decimal = Scalar::decimal(
        DecimalValue::I32(123456),
        DecimalDType::new(10, 0),
        Nullability::NonNullable,
    );

    // Cast to i32 should give exact value
    let result = decimal.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
    assert!(result.is_ok());
    let i32_value: i32 = (&result.unwrap()).try_into().unwrap();
    assert_eq!(i32_value, 123456);
}

#[test]
fn test_native_decimal_type_values_type() {
    // Test DECIMAL_TYPE constant for each type
    assert_eq!(i8::DECIMAL_TYPE, DecimalType::I8);
    assert_eq!(i16::DECIMAL_TYPE, DecimalType::I16);
    assert_eq!(i32::DECIMAL_TYPE, DecimalType::I32);
    assert_eq!(i64::DECIMAL_TYPE, DecimalType::I64);
    assert_eq!(i128::DECIMAL_TYPE, DecimalType::I128);
    assert_eq!(i256::DECIMAL_TYPE, DecimalType::I256);
}

#[test]
fn test_decimal_cast_u64_boundary() {
    // Test U64 boundary case
    let decimal = Scalar::decimal(
        DecimalValue::I128(18446744073709551615_i128), // U64::MAX
        DecimalDType::new(21, 0),
        Nullability::NonNullable,
    );

    let result = decimal.cast(&DType::Primitive(PType::U64, Nullability::NonNullable));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().as_primitive().typed_value::<u64>().unwrap(),
        u64::MAX
    );

    // Test overflow - This value exceeds U64::MAX when cast
    // Note: The cast logic checks the float value against U64::MAX
    let decimal = Scalar::decimal(
        DecimalValue::I128(i128::MAX), // Much larger than U64::MAX
        DecimalDType::new(39, 0),
        Nullability::NonNullable,
    );

    let result = decimal.cast(&DType::Primitive(PType::U64, Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_decimal_i256_overflow_cast() {
    // Test that decimal values too large for i128 are properly handled
    let large_value = i256::from_i128(i128::MAX) + i256::from_i128(1);
    let decimal = Scalar::decimal(
        DecimalValue::I256(large_value),
        DecimalDType::new(40, 0),
        Nullability::NonNullable,
    );

    // This should fail when trying to convert to primitive types
    let result = decimal.cast(&DType::Primitive(PType::I64, Nullability::NonNullable));
    assert!(result.is_err());
}

// Tests for checked_binary_numeric

/// Applies `op` to two non-null operands of `dtype`, returning the result value and its dtype.
fn checked_op(
    lhs: DecimalValue,
    rhs: DecimalValue,
    dtype: DecimalDType,
    op: NumericOperator,
) -> VortexResult<Option<(DecimalValue, DecimalDType)>> {
    let lhs = Scalar::decimal(lhs, dtype, Nullability::NonNullable);
    let rhs = Scalar::decimal(rhs, dtype, Nullability::NonNullable);
    let Some(result) = lhs
        .as_decimal()
        .checked_binary_numeric(&rhs.as_decimal(), op)?
    else {
        return Ok(None);
    };
    let result = result.as_decimal();
    let result_dtype = DecimalDType::try_from(result.dtype())?;
    Ok(result.decimal_value().map(|value| (value, result_dtype)))
}

#[rstest]
#[case::add(
    NumericOperator::Add,
    DecimalDType::new(10, 2),
    Some(DecimalDType::new(11, 2))
)]
#[case::sub(
    NumericOperator::Sub,
    DecimalDType::new(10, 2),
    Some(DecimalDType::new(11, 2))
)]
#[case::mul(
    NumericOperator::Mul,
    DecimalDType::new(10, 2),
    Some(DecimalDType::new(21, 4))
)]
#[case::div(
    NumericOperator::Div,
    DecimalDType::new(10, 2),
    Some(DecimalDType::new(16, 6))
)]
#[case::mul_negative_scale(
    NumericOperator::Mul,
    DecimalDType::new(10, -2),
    Some(DecimalDType::new(21, -4))
)]
#[case::div_negative_scale(
    NumericOperator::Div,
    DecimalDType::new(10, -2),
    Some(DecimalDType::new(12, 2))
)]
#[case::add_saturates_precision(
    NumericOperator::Add,
    DecimalDType::new(MAX_PRECISION, 2),
    Some(DecimalDType::new(MAX_PRECISION, 2))
)]
#[case::mul_saturates_precision(
    NumericOperator::Mul,
    DecimalDType::new(40, 2),
    Some(DecimalDType::new(MAX_PRECISION, 4))
)]
#[case::mul_scale_exceeds_max(NumericOperator::Mul, DecimalDType::new(MAX_PRECISION, 39), None)]
// -70 + -70 is -140, which is not representable as an i8 scale.
#[case::mul_scale_underflows(NumericOperator::Mul, DecimalDType::new(10, -70), None)]
// -64 + -64 is exactly i8::MIN, the most negative scale that is still representable.
#[case::mul_scale_reaches_min(
    NumericOperator::Mul,
    DecimalDType::new(10, -64),
    Some(DecimalDType::new(21, -128))
)]
#[case::div_precision_underflows(NumericOperator::Div, DecimalDType::new(2, -8), None)]
fn test_decimal_numeric_result_dtype(
    #[case] op: NumericOperator,
    #[case] input: DecimalDType,
    #[case] expected: Option<DecimalDType>,
) {
    assert_eq!(decimal_numeric_result_dtype(input, op).ok(), expected);
}

#[rstest]
#[case::add(DecimalDType::new(10, 2), 100, 200, NumericOperator::Add, Some(300))]
#[case::sub(DecimalDType::new(10, 2), 500, 200, NumericOperator::Sub, Some(300))]
// 0.50 * 0.10 = 0.0500, exact at the doubled result scale.
#[case::mul(DecimalDType::new(10, 2), 50, 10, NumericOperator::Mul, Some(500))]
// 0.15 * 0.17 = 0.0255, which a result pinned to scale 2 would have had to round away.
#[case::mul_keeps_every_digit(DecimalDType::new(10, 2), 15, 17, NumericOperator::Mul, Some(255))]
// 10.00 / 0.10 = 100.000000
#[case::div(
    DecimalDType::new(10, 2),
    1_000,
    10,
    NumericOperator::Div,
    Some(100_000_000)
)]
// 10.00 / 3.00 = 3.333333, truncated toward zero at the result scale.
#[case::div_truncates(
    DecimalDType::new(10, 2),
    1_000,
    300,
    NumericOperator::Div,
    Some(3_333_333)
)]
#[case::div_truncates_toward_zero(
    DecimalDType::new(10, 2),
    -1_000,
    300,
    NumericOperator::Div,
    Some(-3_333_333)
)]
#[case::div_by_zero(DecimalDType::new(10, 2), 1_000, 0, NumericOperator::Div, None)]
// 500 * 300 = 150000, stored at scale -4.
#[case::mul_negative_scale(DecimalDType::new(10, -2), 5, 3, NumericOperator::Mul, Some(15))]
// 60000 / 300 = 200.00
#[case::div_negative_scale(DecimalDType::new(10, -2), 600, 3, NumericOperator::Div, Some(20_000))]
// 5e12 / 2e8 = 25000, truncated to the result scale of -4.
#[case::div_negative_result_scale(
    DecimalDType::new(10, -8),
    50_000,
    2,
    NumericOperator::Div,
    Some(2)
)]
fn test_decimal_scalar_checked_binary_numeric(
    #[case] dtype: DecimalDType,
    #[case] lhs: i128,
    #[case] rhs: i128,
    #[case] op: NumericOperator,
    #[case] expected: Option<i128>,
) -> VortexResult<()> {
    let result = checked_op(DecimalValue::I128(lhs), DecimalValue::I128(rhs), dtype, op)?;
    assert_eq!(
        result.map(|(value, _)| value),
        expected.map(DecimalValue::I128)
    );
    Ok(())
}

#[test]
fn test_decimal_scalar_mul_widens_beyond_operand_storage() -> VortexResult<()> {
    // 1e17 * 1e17 needs 35 digits, so widening only to the operands' i64 storage would report a
    // spurious overflow.
    let dtype = DecimalDType::new(18, 0);
    let operand = DecimalValue::I64(100_000_000_000_000_000);

    let (value, result_dtype) = checked_op(operand, operand, dtype, NumericOperator::Mul)?
        .ok_or_else(|| vortex_err!("expected an in-precision product"))?;
    assert_eq!(result_dtype, DecimalDType::new(37, 0));
    assert_eq!(
        value,
        DecimalValue::I128(10_000_000_000_000_000_000_000_000_000_000_000)
    );
    assert_eq!(value.decimal_type(), DecimalType::I128);
    Ok(())
}

#[test]
fn test_decimal_scalar_add_reserves_carry_digit() -> VortexResult<()> {
    // 999 + 2 = 1001 exceeds precision 3, but the result reserves a carry digit for it.
    let dtype = DecimalDType::new(3, 0);

    let (value, result_dtype) = checked_op(
        DecimalValue::I16(999),
        DecimalValue::I16(2),
        dtype,
        NumericOperator::Add,
    )?
    .ok_or_else(|| vortex_err!("expected the carry digit to absorb the result"))?;
    assert_eq!(result_dtype, DecimalDType::new(4, 0));
    assert_eq!(value, DecimalValue::I16(1_001));
    assert_eq!(value.decimal_type(), DecimalType::I16);
    Ok(())
}

#[rstest]
#[case::add(NumericOperator::Add)]
#[case::mul(NumericOperator::Mul)]
fn test_decimal_scalar_overflows_saturated_precision(
    #[case] op: NumericOperator,
) -> VortexResult<()> {
    // At MAX_PRECISION the result precision cannot widen any further, so an out-of-range result
    // is a genuine overflow.
    let dtype = DecimalDType::new(MAX_PRECISION, 0);
    let max = DecimalValue::I256(
        <i256 as NativeDecimalType>::MAX_BY_PRECISION[usize::from(MAX_PRECISION)],
    );

    assert_eq!(checked_op(max, max, dtype, op)?, None);
    Ok(())
}

#[test]
fn test_decimal_scalar_div_reports_unrepresentable_intermediate() -> VortexResult<()> {
    // Div scales the dividend by 10^result_scale, which needs p + result_scale digits. Past
    // MAX_PRECISION there is no wider width to compute in, so 1e79 overflows i256 and the
    // operation reports overflow even though the quotient 1e9 would have fit the result
    // precision. The array kernels size their working width identically.
    let dtype = DecimalDType::new(MAX_PRECISION, 0);
    let pow10 = |exp: u32| {
        i256::from_i128(10)
            .checked_pow(exp)
            .ok_or_else(|| vortex_err!("10^{exp} is representable as i256"))
    };
    let lhs = DecimalValue::I256(pow10(75)?);
    let rhs = DecimalValue::I256(pow10(70)?);

    assert_eq!(
        decimal_numeric_result_dtype(dtype, NumericOperator::Div)?,
        DecimalDType::new(MAX_PRECISION, 4)
    );
    assert_eq!(checked_op(lhs, rhs, dtype, NumericOperator::Div)?, None);
    Ok(())
}

#[test]
fn test_decimal_scalar_null_handling() -> VortexResult<()> {
    let dtype = DecimalDType::new(10, 2);
    let null = Scalar::null(DType::Decimal(dtype, Nullability::Nullable));
    let value = Scalar::decimal(DecimalValue::I64(200), dtype, Nullability::NonNullable);

    let result = null
        .as_decimal()
        .checked_binary_numeric(&value.as_decimal(), NumericOperator::Add)?
        .ok_or_else(|| vortex_err!("a null operand yields a null result, not an overflow"))?;
    assert!(result.is_null());
    assert_eq!(
        result.dtype(),
        &DType::Decimal(DecimalDType::new(11, 2), Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn test_decimal_scalar_mul_rejects_unrepresentable_scale() {
    // 2e70 * 3e70 is 6e140, and a scale of -140 has no i8 representation. Saturating it to -128
    // would have silently returned the raw product 6 as 6e128.
    let dtype = DecimalDType::new(10, -70);
    let lhs = Scalar::decimal(DecimalValue::I64(2), dtype, Nullability::NonNullable);
    let rhs = Scalar::decimal(DecimalValue::I64(3), dtype, Nullability::NonNullable);

    assert!(
        lhs.as_decimal()
            .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Mul)
            .is_err()
    );
}

#[test]
fn test_decimal_scalar_mismatched_decimal_types() {
    let lhs = Scalar::decimal(
        DecimalValue::I64(1),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    let rhs = Scalar::decimal(
        DecimalValue::I64(1),
        DecimalDType::new(10, 3),
        Nullability::NonNullable,
    );

    assert!(
        lhs.as_decimal()
            .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Add)
            .is_err()
    );
}

#[test]
fn test_decimal_value_from_scalar() {
    let value = DecimalValue::I32(12345);
    let scalar = Scalar::from(value);

    // Test extraction
    let extracted: DecimalValue = DecimalValue::try_from(&scalar).unwrap();
    assert_eq!(extracted, value);

    // Test owned extraction
    let extracted_owned: DecimalValue = DecimalValue::try_from(scalar).unwrap();
    assert_eq!(extracted_owned, value);
}

#[test]
fn test_decimal_value_option_from_scalar() {
    // Non-null case
    let value = DecimalValue::I64(999999);
    let scalar = Scalar::from(value);

    let extracted: Option<DecimalValue> = Option::try_from(&scalar).unwrap();
    assert_eq!(extracted, Some(value));

    // Null case
    let null_scalar = Scalar::null(DType::Decimal(
        DecimalDType::new(10, 2),
        Nullability::Nullable,
    ));

    let extracted_null: Option<DecimalValue> = Option::try_from(&null_scalar).unwrap();
    assert_eq!(extracted_null, None);
}

#[test]
fn test_decimal_value_from_conversion() {
    // Test that From<DecimalValue> creates reasonable defaults
    let values = vec![
        DecimalValue::I8(127),
        DecimalValue::I16(32767),
        DecimalValue::I32(1000000),
        DecimalValue::I64(1000000000000),
        DecimalValue::I128(123456789012345678901234567890),
        DecimalValue::I256(i256::from_i128(987654321)),
    ];

    for value in values {
        let scalar = Scalar::from(value);
        assert!(!scalar.is_null());

        // Verify we can extract it back
        let extracted: DecimalValue = DecimalValue::try_from(&scalar).unwrap();
        assert_eq!(extracted, value);
    }
}

#[test]
fn test_decimal_value_checked_add() {
    let a = DecimalValue::I64(100);
    let b = DecimalValue::I64(200);
    let result = a.checked_add(&b).unwrap();
    assert_eq!(result, DecimalValue::I64(300));
}

#[test]
fn test_decimal_value_checked_sub() {
    let a = DecimalValue::I64(500);
    let b = DecimalValue::I64(200);
    let result = a.checked_sub(&b).unwrap();
    assert_eq!(result, DecimalValue::I64(300));
}

#[test]
fn test_decimal_value_checked_mul() {
    let a = DecimalValue::I32(50);
    let b = DecimalValue::I32(10);
    let result = a.checked_mul(&b).unwrap();
    assert_eq!(result, DecimalValue::I32(500));
}

#[test]
fn test_decimal_value_checked_div() {
    let a = DecimalValue::I64(1000);
    let b = DecimalValue::I64(10);
    let result = a.checked_div(&b).unwrap();
    assert_eq!(result, DecimalValue::I64(100));
}

#[test]
fn test_decimal_value_checked_div_by_zero() {
    let a = DecimalValue::I64(1000);
    let b = DecimalValue::I64(0);
    let result = a.checked_div(&b);
    assert_eq!(result, None);
}

#[test]
fn test_decimal_value_mixed_types() {
    // Test operations with different underlying types
    let a = DecimalValue::I8(10);
    let b = DecimalValue::I128(20);
    let result = a.checked_add(&b).unwrap();
    assert_eq!(result, DecimalValue::I128(30));
}

#[test]
fn test_checked_ops_preserve_type() {
    // Operations should return the wider of the two operand types, not unconditionally upcast to I256
    let add = DecimalValue::I32(5)
        .checked_add(&DecimalValue::I32(3))
        .unwrap();
    assert_eq!(add.decimal_type(), DecimalType::I32);

    let sub = DecimalValue::I64(10)
        .checked_sub(&DecimalValue::I64(3))
        .unwrap();
    assert_eq!(sub.decimal_type(), DecimalType::I64);

    let mul = DecimalValue::I8(2)
        .checked_mul(&DecimalValue::I8(3))
        .unwrap();
    assert_eq!(mul.decimal_type(), DecimalType::I8);

    let div = DecimalValue::I128(10)
        .checked_div(&DecimalValue::I128(2))
        .unwrap();
    assert_eq!(div.decimal_type(), DecimalType::I128);

    let add_i256 = DecimalValue::I256(i256::from_i128(1))
        .checked_add(&DecimalValue::I256(i256::from_i128(2)))
        .unwrap();
    assert_eq!(add_i256.decimal_type(), DecimalType::I256);
}

#[test]
fn test_checked_ops_mixed_types_use_wider() {
    let add = DecimalValue::I8(1)
        .checked_add(&DecimalValue::I64(2))
        .unwrap();
    assert_eq!(add.decimal_type(), DecimalType::I64);

    let sub = DecimalValue::I32(10)
        .checked_sub(&DecimalValue::I128(3))
        .unwrap();
    assert_eq!(sub.decimal_type(), DecimalType::I128);
}

#[test]
fn test_checked_ops_overflow_at_target_width() {
    assert_eq!(
        DecimalValue::I8(i8::MAX).checked_add(&DecimalValue::I8(1)),
        None
    );
    assert_eq!(
        DecimalValue::I16(i16::MIN).checked_sub(&DecimalValue::I16(1)),
        None
    );
    assert_eq!(
        DecimalValue::I32(i32::MAX).checked_mul(&DecimalValue::I32(2)),
        None
    );
    assert_eq!(
        DecimalValue::I8(i8::MIN).checked_div(&DecimalValue::I8(-1)),
        None
    );
}

#[test]
fn test_fits_in_precision_exact_boundary() {
    use crate::dtype::DecimalDType;

    // Precision 3 means max value is 10^3 - 1 = 999
    let dtype = DecimalDType::new(3, 0);

    // Test exact upper boundary: 999 should fit
    let value = DecimalValue::I16(999);
    assert!(value.fits_in_precision(dtype));

    // Test just beyond upper boundary: 1000 should NOT fit
    let value = DecimalValue::I16(1000);
    assert!(!value.fits_in_precision(dtype));

    // Test exact lower boundary: -999 should fit
    let value = DecimalValue::I16(-999);
    assert!(value.fits_in_precision(dtype));

    // Test just beyond lower boundary: -1000 should NOT fit
    let value = DecimalValue::I16(-1000);
    assert!(!value.fits_in_precision(dtype));
}

#[test]
fn test_fits_in_precision_zero() {
    use crate::dtype::DecimalDType;

    let dtype = DecimalDType::new(5, 2);

    // Zero should always fit
    let value = DecimalValue::I8(0);
    assert!(value.fits_in_precision(dtype));
}

#[test]
fn test_fits_in_precision_small_precision() {
    use crate::dtype::DecimalDType;

    // Precision 1 means max value is 10^1 - 1 = 9
    let dtype = DecimalDType::new(1, 0);

    // Test values within range
    for i in -9..=9 {
        let value = DecimalValue::I8(i);
        assert!(
            value.fits_in_precision(dtype),
            "value {} should fit in precision 1",
            i
        );
    }

    // Test values outside range
    let value = DecimalValue::I8(10);
    assert!(!value.fits_in_precision(dtype));
    let value = DecimalValue::I8(-10);
    assert!(!value.fits_in_precision(dtype));
}

#[test]
fn test_fits_in_precision_large_precision() {
    use crate::dtype::DecimalDType;

    // Precision 38 means max value is 10^38 - 1
    let dtype = DecimalDType::new(38, 0);

    // Test i128::MAX which is approximately 1.7e38
    // This should NOT fit because 10^38 - 1 < i128::MAX
    let value = DecimalValue::I128(i128::MAX);
    assert!(!value.fits_in_precision(dtype));

    // Test a large value that should fit: 10^37
    let value = DecimalValue::I128(10_i128.pow(37));
    assert!(value.fits_in_precision(dtype));

    // Test 10^38 - 1 (the exact maximum)
    let max_val = i256::from_i128(10).wrapping_pow(38) - i256::from_i128(1);
    let value = DecimalValue::I256(max_val);
    assert!(value.fits_in_precision(dtype));

    // Test 10^38 (just over the maximum)
    let over_max = i256::from_i128(10).wrapping_pow(38);
    let value = DecimalValue::I256(over_max);
    assert!(!value.fits_in_precision(dtype));
}

#[test]
fn test_fits_in_precision_max_precision() {
    use crate::dtype::DecimalDType;

    // Maximum precision is 76
    let dtype = DecimalDType::new(76, 0);

    // Test that reasonable i256 values fit
    let value = DecimalValue::I256(i256::from_i128(i128::MAX));
    assert!(value.fits_in_precision(dtype));

    // Test negative
    let value = DecimalValue::I256(i256::from_i128(i128::MIN));
    assert!(value.fits_in_precision(dtype));
}

#[test]
fn test_fits_in_precision_different_scales() {
    use crate::dtype::DecimalDType;

    // Scale doesn't affect the precision check - it's only about the stored value
    let value = DecimalValue::I32(12345);

    // Precision 5 with different scales
    assert!(value.fits_in_precision(DecimalDType::new(5, 0)));
    assert!(value.fits_in_precision(DecimalDType::new(5, 2)));
    assert!(value.fits_in_precision(DecimalDType::new(5, -2)));

    // Precision 4 should fail (max value 9999, we have 12345)
    assert!(!value.fits_in_precision(DecimalDType::new(4, 0)));
    assert!(!value.fits_in_precision(DecimalDType::new(4, 2)));
}

#[test]
fn test_fits_in_precision_negative_values() {
    use crate::dtype::DecimalDType;

    let dtype = DecimalDType::new(4, 2);

    // Test negative values at boundaries
    // Precision 4 means max magnitude is 9999
    let value = DecimalValue::I16(-9999);
    assert!(value.fits_in_precision(dtype));

    let value = DecimalValue::I16(-10000);
    assert!(!value.fits_in_precision(dtype));

    let value = DecimalValue::I16(-1);
    assert!(value.fits_in_precision(dtype));
}

#[test]
fn test_fits_in_precision_mixed_decimal_value_types() {
    use crate::dtype::DecimalDType;

    let dtype = DecimalDType::new(5, 0);

    // Test that different DecimalValue types work correctly
    assert!(DecimalValue::I8(99).fits_in_precision(dtype));
    assert!(DecimalValue::I16(9999).fits_in_precision(dtype));
    assert!(DecimalValue::I32(99999).fits_in_precision(dtype));
    assert!(!DecimalValue::I64(100000).fits_in_precision(dtype));
    assert!(DecimalValue::I128(99999).fits_in_precision(dtype));
    assert!(!DecimalValue::I256(i256::from_i128(100000)).fits_in_precision(dtype));
}
