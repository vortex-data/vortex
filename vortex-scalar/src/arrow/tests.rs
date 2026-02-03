// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(v2): re-enable tests when removed API features are restored
/*

use arrow_array::Datum;
use rstest::rstest;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::datetime::Date;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::Timestamp;
use vortex_dtype::datetime::TimestampOptions;
use vortex_dtype::extension::EmptyMetadata;

use crate::Scalar;
use crate::tests::Even;

#[test]
fn test_null_scalar_to_arrow() {
    let scalar = Scalar::null(DType::Null);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_bool_scalar_to_arrow() {
    let scalar = Scalar::bool(true, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_null_bool_scalar_to_arrow() {
    let scalar = Scalar::null_typed::<bool>();
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_u8_to_arrow() {
    let scalar = Scalar::primitive(42u8, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_u16_to_arrow() {
    let scalar = Scalar::primitive(1000u16, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_u32_to_arrow() {
    let scalar = Scalar::primitive(100000u32, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_u64_to_arrow() {
    let scalar = Scalar::primitive(10000000000u64, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_i8_to_arrow() {
    let scalar = Scalar::primitive(-42i8, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_i16_to_arrow() {
    let scalar = Scalar::primitive(-1000i16, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_i32_to_arrow() {
    let scalar = Scalar::primitive(-100000i32, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_i64_to_arrow() {
    let scalar = Scalar::primitive(-10000000000i64, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_f16_to_arrow() {
    use vortex_dtype::half::f16;

    let scalar = Scalar::primitive(f16::from_f32(1.234), Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_f32_to_arrow() {
    let scalar = Scalar::primitive(1.234f32, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_primitive_f64_to_arrow() {
    let scalar = Scalar::primitive(1.234567890123f64, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_null_primitive_to_arrow() {
    let scalar = Scalar::null_typed::<i32>();
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_utf8_scalar_to_arrow() {
    let scalar = Scalar::utf8("hello world".to_string(), Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_null_utf8_scalar_to_arrow() {
    let scalar = Scalar::null_typed::<String>();
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_binary_scalar_to_arrow() {
    let data = vec![1u8, 2, 3, 4, 5];
    let scalar = Scalar::binary(data, Nullability::NonNullable);
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_null_binary_scalar_to_arrow() {
    let scalar = Scalar::null(DType::Binary(Nullability::Nullable));
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_decimal_scalars_to_arrow() {
    use vortex_dtype::DecimalDType;

    use crate::decimal::DecimalValue;

    // Test various decimal value types
    let decimal_dtype = DecimalDType::new(5, 2);

    let scalar_i8 = Scalar::decimal(
        DecimalValue::I8(100),
        decimal_dtype,
        Nullability::NonNullable,
    );
    assert!(Arc::<dyn Datum>::try_from(&scalar_i8).is_ok());

    let scalar_i16 = Scalar::decimal(
        DecimalValue::I16(10000),
        decimal_dtype,
        Nullability::NonNullable,
    );
    assert!(Arc::<dyn Datum>::try_from(&scalar_i16).is_ok());

    let scalar_i32 = Scalar::decimal(
        DecimalValue::I32(1000000),
        decimal_dtype,
        Nullability::NonNullable,
    );
    assert!(Arc::<dyn Datum>::try_from(&scalar_i32).is_ok());

    let scalar_i64 = Scalar::decimal(
        DecimalValue::I64(100000000000),
        decimal_dtype,
        Nullability::NonNullable,
    );
    assert!(Arc::<dyn Datum>::try_from(&scalar_i64).is_ok());

    let scalar_i128 = Scalar::decimal(
        DecimalValue::I128(123456789012345678901234567890i128),
        decimal_dtype,
        Nullability::NonNullable,
    );
    assert!(Arc::<dyn Datum>::try_from(&scalar_i128).is_ok());

    // Test i256
    use vortex_dtype::i256;
    let value_i256 = i256::from_i128(123456789012345678901234567890i128);
    let scalar_i256 = Scalar::decimal(
        DecimalValue::I256(value_i256),
        decimal_dtype,
        Nullability::NonNullable,
    );
    assert!(Arc::<dyn Datum>::try_from(&scalar_i256).is_ok());
}

#[test]
fn test_null_decimal_to_arrow() {
    use vortex_dtype::DecimalDType;

    let decimal_dtype = DecimalDType::new(10, 2);
    let scalar = Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable));
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
#[should_panic(expected = "struct scalar conversion")]
fn test_struct_scalar_to_arrow_todo() {
    use vortex_dtype::FieldDType;
    use vortex_dtype::StructFields;

    let struct_dtype = DType::Struct(
        StructFields::from_iter([(
            "field1",
            FieldDType::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
        )]),
        Nullability::NonNullable,
    );

    let struct_scalar = Scalar::struct_(
        struct_dtype,
        vec![Scalar::primitive(42i32, Nullability::NonNullable)],
    );
    Arc::<dyn Datum>::try_from(&struct_scalar).unwrap();
}

#[test]
#[should_panic(expected = "list scalar conversion")]
fn test_list_scalar_to_arrow_todo() {
    let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
    let list_scalar = Scalar::list(
        element_dtype,
        vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ],
        Nullability::NonNullable,
    );

    Arc::<dyn Datum>::try_from(&list_scalar).unwrap();
}

#[test]
#[should_panic(expected = "Cannot convert extension scalar")]
fn test_non_temporal_extension_to_arrow_todo() {
    let scalar =
        Scalar::extension::<Even>(EmptyMetadata, Some(32), Nullability::NonNullable).unwrap();
    Arc::<dyn Datum>::try_from(&scalar).unwrap();
}

#[rstest]
#[case(TimeUnit::Nanoseconds, jiff::civil::Time::MIN)]
#[case(TimeUnit::Microseconds, jiff::civil::Time::MIN)]
#[case(TimeUnit::Milliseconds, jiff::civil::Time::MIN)]
#[case(TimeUnit::Seconds, jiff::civil::Time::MIN)]
fn test_temporal_time_to_arrow(#[case] time_unit: TimeUnit, #[case] value: jiff::civil::Time) {
    let scalar =
        Scalar::extension::<Time>(time_unit, Some(value), Nullability::NonNullable).unwrap();
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[rstest]
#[case(TimeUnit::Milliseconds, jiff::civil::Date::new(2023, 1, 1).unwrap())]
#[case(TimeUnit::Days, jiff::civil::Date::new(2023, 1, 1).unwrap())]
fn test_temporal_date_to_arrow(#[case] time_unit: TimeUnit, #[case] value: jiff::civil::Date) {
    let scalar =
        Scalar::extension::<Date>(time_unit, Some(value), Nullability::NonNullable).unwrap();
    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[rstest]
#[case(TimeUnit::Nanoseconds)]
#[case(TimeUnit::Microseconds)]
#[case(TimeUnit::Milliseconds)]
#[case(TimeUnit::Seconds)]
fn test_temporal_timestamp_to_arrow(#[case] time_unit: TimeUnit) {
    let scalar = Scalar::extension::<Timestamp>(
        TimestampOptions {
            unit: time_unit,
            tz: None,
        },
        Some(TimestampValue::Unzoned(jiff::Timestamp::UNIX_EPOCH)),
        Nullability::NonNullable,
    )
    .unwrap();

    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[rstest]
#[case(TimeUnit::Nanoseconds, "UTC")]
#[case(TimeUnit::Microseconds, "EST")]
#[case(TimeUnit::Milliseconds, "ABC")]
#[case(TimeUnit::Seconds, "UTC")]
fn test_temporal_timestamp_tz_to_arrow(#[case] time_unit: TimeUnit, #[case] tz: &str) {
    let scalar = Scalar::extension::<Timestamp>(
        TimestampOptions {
            unit: time_unit,
            tz: Some(tz.into()),
        },
        Some(TimestampValue::Zoned(
            jiff::Timestamp::UNIX_EPOCH.in_tz(tz).unwrap(),
        )),
        Nullability::NonNullable,
    )
    .unwrap();

    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_temporal_with_null_value() {
    let scalar =
        Scalar::extension::<Time>(TimeUnit::Milliseconds, None, Nullability::Nullable).unwrap();

    let _result = Arc::<dyn Datum>::try_from(&scalar).unwrap();
}
*/
