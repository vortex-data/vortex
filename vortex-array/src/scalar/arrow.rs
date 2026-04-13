// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions between [`Scalar`] and Arrow scalar types.

use std::sync::Arc;

use arrow_array::Scalar as ArrowScalar;
use arrow_array::*;
use vortex_error::VortexError;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;
use crate::scalar::BinaryScalar;
use crate::scalar::BoolScalar;
use crate::scalar::DecimalScalar;
use crate::scalar::DecimalValue;
use crate::scalar::ExtScalar;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::scalar::Utf8Scalar;

/// Arrow represents scalars as single-element arrays. This constant is the length of those arrays.
const SCALAR_ARRAY_LEN: usize = 1;

/// Converts an optional value to an Arrow scalar array.
macro_rules! value_to_arrow_scalar {
    ($V:expr, $AR:ty) => {
        Ok(std::sync::Arc::new(
            $V.map(<$AR>::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(<$AR>::new_null(SCALAR_ARRAY_LEN))),
        ))
    };
}

/// Converts an optional timestamp value to an Arrow scalar array.
macro_rules! timestamp_to_arrow_scalar {
    ($V:expr, $TZ:expr, $AR:ty) => {{
        let array = match $V {
            Some(v) => <$AR>::new_scalar(v).into_inner(),
            None => <$AR>::new_null(SCALAR_ARRAY_LEN),
        }
        .with_timezone_opt($TZ);
        Ok(Arc::new(ArrowScalar::new(array)))
    }};
}

impl TryFrom<&Scalar> for Arc<dyn Datum> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Arc<dyn Datum>, Self::Error> {
        match value.dtype() {
            DType::Null => Ok(Arc::new(NullArray::new(SCALAR_ARRAY_LEN))),
            DType::Bool(_) => bool_to_arrow(value.as_bool()),
            DType::Primitive(..) => primitive_to_arrow(value.as_primitive()),
            DType::Decimal(..) => decimal_to_arrow(value.as_decimal()),
            DType::Utf8(_) => utf8_to_arrow(value.as_utf8()),
            DType::Binary(_) => binary_to_arrow(value.as_binary()),
            DType::Struct(..) => unimplemented!("struct scalar conversion"),
            DType::List(..) => unimplemented!("list scalar conversion"),
            DType::FixedSizeList(..) => unimplemented!("fixed-size list scalar conversion"),
            DType::Extension(..) => extension_to_arrow(value.as_extension()),
            DType::Variant(_) => unimplemented!("Variant scalar conversion"),
        }
    }
}

/// Convert a [`BoolScalar`] to an Arrow [`Datum`].
fn bool_to_arrow(scalar: BoolScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    value_to_arrow_scalar!(scalar.value(), BooleanArray)
}

/// Convert a [`PrimitiveScalar`] to an Arrow [`Datum`].
fn primitive_to_arrow(scalar: PrimitiveScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    match scalar.ptype() {
        PType::U8 => value_to_arrow_scalar!(scalar.typed_value(), UInt8Array),
        PType::U16 => value_to_arrow_scalar!(scalar.typed_value(), UInt16Array),
        PType::U32 => value_to_arrow_scalar!(scalar.typed_value(), UInt32Array),
        PType::U64 => value_to_arrow_scalar!(scalar.typed_value(), UInt64Array),
        PType::I8 => value_to_arrow_scalar!(scalar.typed_value(), Int8Array),
        PType::I16 => value_to_arrow_scalar!(scalar.typed_value(), Int16Array),
        PType::I32 => value_to_arrow_scalar!(scalar.typed_value(), Int32Array),
        PType::I64 => value_to_arrow_scalar!(scalar.typed_value(), Int64Array),
        PType::F16 => value_to_arrow_scalar!(scalar.typed_value(), Float16Array),
        PType::F32 => value_to_arrow_scalar!(scalar.typed_value(), Float32Array),
        PType::F64 => value_to_arrow_scalar!(scalar.typed_value(), Float64Array),
    }
}

/// Convert a [`DecimalScalar`] to an Arrow [`Datum`].
fn decimal_to_arrow(scalar: DecimalScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    // TODO(joe): Replace with decimal32, etc. once Arrow supports them.
    match scalar.decimal_value() {
        Some(DecimalValue::I8(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
        Some(DecimalValue::I16(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
        Some(DecimalValue::I32(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
        Some(DecimalValue::I64(v)) => Ok(Arc::new(Decimal128Array::new_scalar(v as i128))),
        Some(DecimalValue::I128(v128)) => Ok(Arc::new(Decimal128Array::new_scalar(v128))),
        Some(DecimalValue::I256(v256)) => Ok(Arc::new(Decimal256Array::new_scalar(v256.into()))),
        None => Ok(Arc::new(arrow_array::Scalar::new(
            Decimal128Array::new_null(SCALAR_ARRAY_LEN),
        ))),
    }
}

/// Convert a [`Utf8Scalar`] to an Arrow [`Datum`].
fn utf8_to_arrow(scalar: Utf8Scalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    value_to_arrow_scalar!(scalar.value(), StringViewArray)
}

/// Convert a [`BinaryScalar`] to an Arrow [`Datum`].
fn binary_to_arrow(scalar: BinaryScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    value_to_arrow_scalar!(scalar.value(), BinaryViewArray)
}

/// Convert an [`ExtScalar`] to an Arrow [`Datum`].
///
/// Currently only temporal extension types (timestamps, dates, and times) are supported.
fn extension_to_arrow(scalar: ExtScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    let ext_dtype = scalar.ext_dtype();
    let Some(temporal) = ext_dtype.metadata_opt::<AnyTemporal>() else {
        vortex_bail!(
            "Cannot convert extension scalar {} to Arrow",
            ext_dtype.id()
        )
    };

    let storage_scalar = scalar.to_storage_scalar();
    let primitive = storage_scalar
        .as_primitive_opt()
        .ok_or_else(|| vortex_err!("Expected primitive scalar"))?;

    match temporal {
        TemporalMetadata::Timestamp(unit, tz) => {
            let value = primitive.as_::<i64>();
            match unit {
                TimeUnit::Nanoseconds => {
                    timestamp_to_arrow_scalar!(value, tz.clone(), TimestampNanosecondArray)
                }
                TimeUnit::Microseconds => {
                    timestamp_to_arrow_scalar!(value, tz.clone(), TimestampMicrosecondArray)
                }
                TimeUnit::Milliseconds => {
                    timestamp_to_arrow_scalar!(value, tz.clone(), TimestampMillisecondArray)
                }
                TimeUnit::Seconds => {
                    timestamp_to_arrow_scalar!(value, tz.clone(), TimestampSecondArray)
                }
                TimeUnit::Days => {
                    vortex_bail!("Unsupported TimeUnit {unit} for {}", ext_dtype.id())
                }
            }
        }
        TemporalMetadata::Date(unit) => match unit {
            TimeUnit::Milliseconds => {
                value_to_arrow_scalar!(primitive.as_::<i64>(), Date64Array)
            }
            TimeUnit::Days => {
                value_to_arrow_scalar!(primitive.as_::<i32>(), Date32Array)
            }
            TimeUnit::Nanoseconds | TimeUnit::Microseconds | TimeUnit::Seconds => {
                vortex_bail!("Unsupported TimeUnit {unit} for {}", ext_dtype.id())
            }
        },
        TemporalMetadata::Time(unit) => match unit {
            TimeUnit::Nanoseconds => {
                value_to_arrow_scalar!(primitive.as_::<i64>(), Time64NanosecondArray)
            }
            TimeUnit::Microseconds => {
                value_to_arrow_scalar!(primitive.as_::<i64>(), Time64MicrosecondArray)
            }
            TimeUnit::Milliseconds => {
                value_to_arrow_scalar!(primitive.as_::<i32>(), Time32MillisecondArray)
            }
            TimeUnit::Seconds => {
                value_to_arrow_scalar!(primitive.as_::<i32>(), Time32SecondArray)
            }
            TimeUnit::Days => {
                vortex_bail!("Unsupported TimeUnit {unit} for {}", ext_dtype.id())
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Datum;
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldDType;
    use crate::dtype::NativeDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::i256;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::extension::datetime::TimestampOptions;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

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
        let scalar = Scalar::null(bool::dtype().as_nullable());
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
        use crate::dtype::half::f16;

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
        let scalar = Scalar::null(i32::dtype().as_nullable());
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
        let scalar = Scalar::null(String::dtype().as_nullable());
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
            DecimalValue::I32(99999),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert!(Arc::<dyn Datum>::try_from(&scalar_i32).is_ok());

        let scalar_i64 = Scalar::decimal(
            DecimalValue::I64(99999),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert!(Arc::<dyn Datum>::try_from(&scalar_i64).is_ok());

        let scalar_i128 = Scalar::decimal(
            DecimalValue::I128(99999),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert!(Arc::<dyn Datum>::try_from(&scalar_i128).is_ok());

        // Test i256

        let value_i256 = i256::from_i128(99999);
        let scalar_i256 = Scalar::decimal(
            DecimalValue::I256(value_i256),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert!(Arc::<dyn Datum>::try_from(&scalar_i256).is_ok());
    }

    #[test]
    fn test_null_decimal_to_arrow() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let scalar = Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable));
        let result = Arc::<dyn Datum>::try_from(&scalar);
        assert!(result.is_ok());
    }

    #[test]
    #[should_panic(expected = "struct scalar conversion")]
    fn test_struct_scalar_to_arrow_todo() {
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
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        struct SomeExt;
        impl ExtVTable for SomeExt {
            type Metadata = String;
            type NativeValue<'a> = &'a str;

            fn id(&self) -> ExtId {
                ExtId::new("some_ext")
            }

            fn serialize_metadata(&self, _options: &Self::Metadata) -> VortexResult<Vec<u8>> {
                vortex_bail!("not implemented")
            }

            fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
                vortex_bail!("not implemented")
            }

            fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
                Ok(())
            }

            fn unpack_native<'a>(
                _ext_dtype: &'a ExtDType<Self>,
                _storage_value: &'a ScalarValue,
            ) -> VortexResult<Self::NativeValue<'a>> {
                Ok("")
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
    #[case(TimeUnit::Microseconds, "Asia/Qatar", 1234567890000000i64)]
    #[case(TimeUnit::Microseconds, "Australia/Sydney", 1234567890000000i64)]
    #[case(TimeUnit::Milliseconds, "HST", 1234567890000i64)]
    #[case(TimeUnit::Seconds, "GMT", 1234567890i64)]
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
    #[should_panic(expected = "DType utf8 is not a primitive type")]
    fn test_temporal_non_primitive_storage_error() {
        let _scalar = Scalar::extension::<Time>(
            TimeUnit::Nanoseconds,
            Scalar::utf8("not a timestamp", Nullability::NonNullable),
        );
    }
}
