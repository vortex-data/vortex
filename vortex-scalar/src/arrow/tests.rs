// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

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
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Scalar;

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
    use vortex_dtype::ExtID;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct SomeExt;
    impl ExtDTypeVTable for SomeExt {
        type Metadata = String;

        fn id(&self) -> ExtID {
            ExtID::new_ref("some_ext")
        }

        fn serialize(&self, _options: &Self::Metadata) -> VortexResult<Vec<u8>> {
            vortex_bail!("not implemented")
        }

        fn deserialize(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            vortex_bail!("not implemented")
        }

        fn validate_dtype(
            &self,
            _options: &Self::Metadata,
            _storage_dtype: &DType,
        ) -> VortexResult<()> {
            Ok(())
        }
    }

    let scalar = Scalar::extension::<SomeExt>(
        "".into(),
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    Arc::<dyn Datum>::try_from(&scalar).unwrap();
}

#[rstest]
#[case(TimeUnit::Nanoseconds, PType::I64, 123456789i64)]
#[case(TimeUnit::Microseconds, PType::I64, 123456789i64)]
#[case(TimeUnit::Milliseconds, PType::I32, 123456i64)]
#[case(TimeUnit::Seconds, PType::I32, 1234i64)]
fn test_temporal_time_to_arrow(
    #[case] time_unit: TimeUnit,
    #[case] ptype: PType,
    #[case] value: i64,
) {
    let scalar = Scalar::extension::<Time>(
        time_unit,
        match ptype {
            PType::I32 => {
                let i32_value = i32::try_from(value).expect("test value should fit in i32");
                Scalar::primitive(i32_value, Nullability::NonNullable)
            }
            PType::I64 => Scalar::primitive(value, Nullability::NonNullable),
            _ => unreachable!(),
        },
    );

    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[rstest]
#[case(TimeUnit::Milliseconds, PType::I64, 1234567890000i64)]
#[case(TimeUnit::Days, PType::I32, 19000i64)]
fn test_temporal_date_to_arrow(
    #[case] time_unit: TimeUnit,
    #[case] ptype: PType,
    #[case] value: i64,
) {
    let scalar = Scalar::extension::<Date>(
        time_unit,
        match ptype {
            PType::I32 => {
                let i32_value = i32::try_from(value).expect("test value should fit in i32");
                Scalar::primitive(i32_value, Nullability::NonNullable)
            }
            PType::I64 => Scalar::primitive(value, Nullability::NonNullable),
            _ => unreachable!(),
        },
    );

    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[rstest]
#[case(TimeUnit::Nanoseconds, 1234567890000000000i64)]
#[case(TimeUnit::Microseconds, 1234567890000000i64)]
#[case(TimeUnit::Milliseconds, 1234567890000i64)]
#[case(TimeUnit::Seconds, 1234567890i64)]
fn test_temporal_timestamp_to_arrow(#[case] time_unit: TimeUnit, #[case] value: i64) {
    let scalar = Scalar::extension::<Timestamp>(
        TimestampOptions {
            unit: time_unit,
            tz: None,
        },
        Scalar::primitive(value, Nullability::NonNullable),
    );

    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[rstest]
#[case(TimeUnit::Nanoseconds, "UTC", 1234567890000000000i64)]
#[case(TimeUnit::Microseconds, "EST", 1234567890000000i64)]
#[case(TimeUnit::Milliseconds, "ABC", 1234567890000i64)]
#[case(TimeUnit::Seconds, "UTC", 1234567890i64)]
fn test_temporal_timestamp_tz_to_arrow(
    #[case] time_unit: TimeUnit,
    #[case] tz: &str,
    #[case] value: i64,
) {
    let scalar = Scalar::extension::<Timestamp>(
        TimestampOptions {
            unit: time_unit,
            tz: Some(tz.into()),
        },
        Scalar::primitive(value, Nullability::NonNullable),
    );

    let result = Arc::<dyn Datum>::try_from(&scalar);
    assert!(result.is_ok());
}

#[test]
fn test_temporal_with_null_value() {
    let scalar = Scalar::extension::<Time>(
        TimeUnit::Milliseconds,
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
    );

    let _result = Arc::<dyn Datum>::try_from(&scalar).unwrap();
}

#[test]
#[should_panic(expected = "DType is not a primitive type")]
fn test_temporal_non_primitive_storage_error() {
    let _scalar = Scalar::extension::<Time>(
        TimeUnit::Nanoseconds,
        Scalar::utf8("not a timestamp", Nullability::NonNullable),
    );
}
