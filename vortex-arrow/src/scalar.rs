// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions between [`Scalar`] and Arrow scalar types.

use std::sync::Arc;

use arrow_array::Scalar as ArrowScalar;
use arrow_array::*;
use arrow_buffer::NullBuffer;
use arrow_buffer::OffsetBuffer;
use arrow_schema::Field;
use arrow_schema::Fields;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::extension::datetime::TemporalMetadata;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::BinaryScalar;
use vortex_array::scalar::BoolScalar;
use vortex_array::scalar::DecimalScalar;
use vortex_array::scalar::ExtScalar;
use vortex_array::scalar::MapScalar;
use vortex_array::scalar::PrimitiveScalar;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::Utf8Scalar;
use vortex_error::VortexError;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::dtype::to_data_type_naive;

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

/// Convert a Vortex [`Scalar`] into an Arrow [`Datum`] (a single-element Arrow array).
///
/// This mirrors a `TryFrom<&Scalar> for Arc<dyn Datum>` conversion; a separate trait is
/// required because both `Scalar` and `Datum` are foreign to this crate.
pub trait ToArrowDatum {
    /// Convert this scalar to an Arrow [`Datum`].
    fn to_arrow_datum(&self) -> Result<Arc<dyn Datum>, VortexError>;
}

impl ToArrowDatum for Scalar {
    fn to_arrow_datum(&self) -> Result<Arc<dyn Datum>, VortexError> {
        let value = self;
        match value.dtype() {
            DType::Null => Ok(Arc::new(NullArray::new(SCALAR_ARRAY_LEN))),
            DType::Bool(_) => bool_to_arrow(value.as_bool()),
            DType::Primitive(..) => primitive_to_arrow(value.as_primitive()),
            DType::Decimal(..) => decimal_to_arrow(value.as_decimal()),
            DType::Utf8(_) => utf8_to_arrow(value.as_utf8()),
            DType::Binary(_) => binary_to_arrow(value.as_binary()),
            DType::List(..) => vortex_bail!("list scalar conversion is not supported"),
            DType::FixedSizeList(..) => {
                vortex_bail!("fixed-size list scalar conversion is not supported")
            }
            DType::Map(..) => map_to_arrow(value.as_map()),
            DType::Struct(..) => vortex_bail!("struct scalar conversion is not supported"),
            DType::Union(..) => vortex_bail!("union scalar conversion is not supported"),
            DType::Variant(_) => vortex_bail!("Variant scalar conversion is not supported"),
            DType::Extension(..) => extension_to_arrow(value.as_extension()),
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
    let DType::Decimal(decimal_dtype, _) = scalar.dtype() else {
        vortex_bail!("Expected decimal scalar, got {}", scalar.dtype());
    };
    let precision = decimal_dtype.precision();
    let scale = decimal_dtype.scale();
    // TODO(joe): Replace with decimal32, etc. once Arrow supports them.
    match scalar.decimal_value() {
        Some(value) => {
            let value = value.as_i256();
            if precision <= 38 {
                let value = value.maybe_i128().ok_or_else(|| {
                    vortex_err!(
                        "Decimal value {value} cannot fit in Arrow Decimal128 for precision {precision}"
                    )
                })?;
                decimal128_scalar(value, precision, scale)
            } else {
                decimal256_scalar(value, precision, scale)
            }
        }
        None => {
            let data_type = to_data_type_naive(scalar.dtype())?;
            Ok(Arc::new(ArrowScalar::new(new_null_array(
                &data_type,
                SCALAR_ARRAY_LEN,
            ))))
        }
    }
}

fn decimal128_scalar(value: i128, precision: u8, scale: i8) -> Result<Arc<dyn Datum>, VortexError> {
    let array = Decimal128Array::new_scalar(value)
        .into_inner()
        .with_precision_and_scale(precision, scale)?;
    Ok(Arc::new(ArrowScalar::new(array)))
}

fn decimal256_scalar(value: i256, precision: u8, scale: i8) -> Result<Arc<dyn Datum>, VortexError> {
    let array = Decimal256Array::new_scalar(value.into())
        .into_inner()
        .with_precision_and_scale(precision, scale)?;
    Ok(Arc::new(ArrowScalar::new(array)))
}

/// Convert a [`Utf8Scalar`] to an Arrow [`Datum`].
fn utf8_to_arrow(scalar: Utf8Scalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    value_to_arrow_scalar!(scalar.value(), StringViewArray)
}

/// Convert a [`BinaryScalar`] to an Arrow [`Datum`].
fn binary_to_arrow(scalar: BinaryScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    value_to_arrow_scalar!(scalar.value(), BinaryViewArray)
}

/// Convert a [`MapScalar`] to an Arrow [`Datum`].
fn map_to_arrow(scalar: MapScalar<'_>) -> Result<Arc<dyn Datum>, VortexError> {
    let map_dtype = scalar.map_dtype();
    let key_dtype = map_dtype.key_dtype();
    let value_dtype = map_dtype.value_dtype();
    let key_field = Field::new("key", to_data_type_naive(&key_dtype)?, false);
    let value_field = Field::new(
        "value",
        to_data_type_naive(&value_dtype)?,
        value_dtype.is_nullable(),
    );
    let fields = Fields::from(vec![key_field, value_field]);

    let entries = scalar.entries().collect::<Vec<_>>();
    let keys = entries
        .iter()
        .map(|(key, _)| key.to_arrow_datum())
        .collect::<Result<Vec<_>, _>>()?;
    let values = entries
        .iter()
        .map(|(_, value)| value.to_arrow_datum())
        .collect::<Result<Vec<_>, _>>()?;

    let key_array = concat_scalar_arrays(&keys, &key_dtype)?;
    let value_array = concat_scalar_arrays(&values, &value_dtype)?;
    let entries = StructArray::try_new_with_length(
        fields.clone(),
        vec![key_array, value_array],
        None,
        entries.len(),
    )?;

    let entries_len = entries.len();
    let entries_len = i32::try_from(entries_len).map_err(|_| {
        vortex_err!(
            "Cannot convert map scalar with {entries_len} entries to Arrow: MapArray offsets are i32"
        )
    })?;
    let offsets = OffsetBuffer::new(vec![0_i32, entries_len].into());
    let entries_field = Arc::new(Field::new_struct("entries", fields, false));
    let nulls = scalar
        .is_null()
        .then(|| NullBuffer::new_null(SCALAR_ARRAY_LEN));
    let map = MapArray::try_new(
        entries_field,
        offsets,
        entries,
        nulls,
        map_dtype.keys_sorted(),
    )?;
    Ok(Arc::new(ArrowScalar::new(map)))
}

fn concat_scalar_arrays(
    scalars: &[Arc<dyn Datum>],
    dtype: &DType,
) -> Result<ArrayRef, VortexError> {
    if scalars.is_empty() {
        return Ok(new_empty_array(&to_data_type_naive(dtype)?));
    }

    let arrays = scalars
        .iter()
        .map(|scalar| scalar.get().0)
        .collect::<Vec<_>>();
    Ok(arrow_select::concat::concat(&arrays)?)
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

    use arrow_array::Array;
    use arrow_array::Decimal128Array;
    use arrow_array::Int32Array;
    use arrow_array::MapArray;
    use arrow_array::StringViewArray;
    use arrow_schema::DataType;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::FieldDType;
    use vortex_array::dtype::NativeDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtId;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::dtype::i256;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::Time;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::extension::datetime::TimestampOptions;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::ToArrowDatum;

    #[test]
    fn test_null_scalar_to_arrow() {
        let scalar = Scalar::null(DType::Null);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_bool_scalar_to_arrow() {
        let scalar = Scalar::bool(true, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_bool_scalar_to_arrow() {
        let scalar = Scalar::null(bool::dtype().as_nullable());
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_u8_to_arrow() {
        let scalar = Scalar::primitive(42u8, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_u16_to_arrow() {
        let scalar = Scalar::primitive(1000u16, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_u32_to_arrow() {
        let scalar = Scalar::primitive(100000u32, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_u64_to_arrow() {
        let scalar = Scalar::primitive(10000000000u64, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_i8_to_arrow() {
        let scalar = Scalar::primitive(-42i8, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_i16_to_arrow() {
        let scalar = Scalar::primitive(-1000i16, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_i32_to_arrow() {
        let scalar = Scalar::primitive(-100000i32, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_i64_to_arrow() {
        let scalar = Scalar::primitive(-10000000000i64, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_f16_to_arrow() {
        use vortex_array::dtype::half::f16;

        let scalar = Scalar::primitive(f16::from_f32(1.234), Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_f32_to_arrow() {
        let scalar = Scalar::primitive(1.234f32, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_f64_to_arrow() {
        let scalar = Scalar::primitive(1.234567890123f64, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_primitive_to_arrow() {
        let scalar = Scalar::null(i32::dtype().as_nullable());
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_scalar_to_arrow() {
        let scalar = Scalar::utf8("hello world".to_string(), Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_utf8_scalar_to_arrow() {
        let scalar = Scalar::null(String::dtype().as_nullable());
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_binary_scalar_to_arrow() {
        let data = vec![1u8, 2, 3, 4, 5];
        let scalar = Scalar::binary(data, Nullability::NonNullable);
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_binary_scalar_to_arrow() {
        let scalar = Scalar::null(DType::Binary(Nullability::Nullable));
        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    fn assert_arrow_scalar_data_type(scalar: &Scalar, expected: DataType) -> VortexResult<()> {
        let datum = scalar.to_arrow_datum()?;
        let (array, is_scalar) = datum.get();
        assert!(is_scalar);
        assert_eq!(array.data_type(), &expected);
        Ok(())
    }

    #[test]
    fn test_decimal_scalars_to_arrow() -> VortexResult<()> {
        // Test various decimal value types
        let decimal_dtype = DecimalDType::new(5, 2);

        let scalar_i8 = Scalar::decimal(
            DecimalValue::I8(100),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert_arrow_scalar_data_type(&scalar_i8, DataType::Decimal128(5, 2))?;

        let scalar_i16 = Scalar::decimal(
            DecimalValue::I16(10000),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert_arrow_scalar_data_type(&scalar_i16, DataType::Decimal128(5, 2))?;

        let scalar_i32 = Scalar::decimal(
            DecimalValue::I32(99999),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert_arrow_scalar_data_type(&scalar_i32, DataType::Decimal128(5, 2))?;

        let scalar_i64 = Scalar::decimal(
            DecimalValue::I64(99999),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert_arrow_scalar_data_type(&scalar_i64, DataType::Decimal128(5, 2))?;

        let scalar_i128 = Scalar::decimal(
            DecimalValue::I128(99999),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert_arrow_scalar_data_type(&scalar_i128, DataType::Decimal128(5, 2))?;

        let value_i256 = i256::from_i128(99999);
        let scalar_i256 = Scalar::decimal(
            DecimalValue::I256(value_i256),
            decimal_dtype,
            Nullability::NonNullable,
        );
        assert_arrow_scalar_data_type(&scalar_i256, DataType::Decimal128(5, 2))?;

        Ok(())
    }

    #[test]
    fn decimal_i64_with_wide_precision_exports_decimal256() -> VortexResult<()> {
        let scalar = Scalar::decimal(
            DecimalValue::I64(1),
            DecimalDType::new(39, 0),
            Nullability::NonNullable,
        );

        assert_arrow_scalar_data_type(&scalar, DataType::Decimal256(39, 0))
    }

    #[test]
    fn decimal_i256_with_narrow_precision_exports_decimal128() -> VortexResult<()> {
        let scalar = Scalar::decimal(
            DecimalValue::I256(i256::from_i128(1234)),
            DecimalDType::new(4, 2),
            Nullability::NonNullable,
        );

        assert_arrow_scalar_data_type(&scalar, DataType::Decimal128(4, 2))
    }

    #[test]
    fn test_null_decimal_to_arrow() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let scalar = Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable));
        assert_arrow_scalar_data_type(&scalar, DataType::Decimal128(10, 2))?;

        let decimal_dtype = DecimalDType::new(39, 2);
        let scalar = Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable));
        assert_arrow_scalar_data_type(&scalar, DataType::Decimal256(39, 2))
    }

    #[test]
    fn decimal_scalar_to_arrow_preserves_precision_and_scale() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(12, 3);
        let scalar = Scalar::decimal(
            DecimalValue::I128(12345),
            decimal_dtype,
            Nullability::NonNullable,
        );

        let datum = scalar.to_arrow_datum()?;
        let (array, is_scalar) = datum.get();
        assert!(is_scalar);
        let decimal = array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal scalar should convert to Decimal128");
        assert_eq!(decimal.precision(), 12);
        assert_eq!(decimal.scale(), 3);

        Ok(())
    }

    #[test]
    fn test_map_scalar_to_arrow() -> VortexResult<()> {
        let dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
            Nullability::Nullable,
        )?;
        let scalar = Scalar::try_map(
            dtype,
            [
                (
                    Scalar::primitive(1i32, Nullability::NonNullable),
                    Scalar::utf8("one", Nullability::Nullable),
                ),
                (
                    Scalar::primitive(2i32, Nullability::NonNullable),
                    Scalar::null(DType::Utf8(Nullability::Nullable)),
                ),
            ],
        )?;

        let datum = scalar.to_arrow_datum()?;
        let (array, is_scalar) = datum.get();
        assert!(is_scalar);
        let map = array
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("map scalar should convert to MapArray");
        assert_eq!(map.len(), 1);
        assert_eq!(map.value_offsets(), &[0, 2]);
        assert!(map.is_valid(0));
        assert_eq!(
            map.keys()
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("map key array should be Int32")
                .values(),
            &[1, 2]
        );
        let values = map
            .values()
            .as_any()
            .downcast_ref::<StringViewArray>()
            .expect("map value array should be StringView");
        assert_eq!(values.value(0), "one");
        assert!(values.is_null(1));

        Ok(())
    }

    #[test]
    fn map_decimal_scalar_to_arrow_preserves_decimal_type() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(9, 2);
        let dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Decimal(decimal_dtype, Nullability::Nullable),
            false,
            Nullability::NonNullable,
        )?;
        let scalar = Scalar::try_map(
            dtype,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::decimal(
                    DecimalValue::I256(i256::from_i128(12345)),
                    decimal_dtype,
                    Nullability::Nullable,
                ),
            )],
        )?;

        let datum = scalar.to_arrow_datum()?;
        let (array, is_scalar) = datum.get();
        assert!(is_scalar);
        let map = array
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("map scalar should convert to MapArray");
        assert_eq!(map.values().data_type(), &DataType::Decimal128(9, 2));

        let wide_decimal_dtype = DecimalDType::new(39, 2);
        let dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Decimal(wide_decimal_dtype, Nullability::Nullable),
            false,
            Nullability::NonNullable,
        )?;
        let scalar = Scalar::try_map(
            dtype,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::decimal(
                    DecimalValue::I64(12345),
                    wide_decimal_dtype,
                    Nullability::Nullable,
                ),
            )],
        )?;

        let datum = scalar.to_arrow_datum()?;
        let (array, is_scalar) = datum.get();
        assert!(is_scalar);
        let map = array
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("map scalar should convert to MapArray");
        assert_eq!(map.values().data_type(), &DataType::Decimal256(39, 2));

        Ok(())
    }

    #[test]
    fn map_scalar_with_unsupported_nested_value_errors_without_panic() -> VortexResult<()> {
        let struct_dtype = DType::Struct(
            StructFields::from_iter([(
                "field1",
                FieldDType::from(DType::Primitive(PType::I32, Nullability::NonNullable)),
            )]),
            Nullability::NonNullable,
        );
        let dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            struct_dtype.clone(),
            false,
            Nullability::NonNullable,
        )?;
        let scalar = Scalar::try_map(
            dtype,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::struct_(
                    struct_dtype,
                    vec![Scalar::primitive(42i32, Nullability::NonNullable)],
                ),
            )],
        )?;

        let error = scalar
            .to_arrow_datum()
            .err()
            .expect("unsupported nested map value should error");
        assert!(error.to_string().contains("struct scalar conversion"));

        Ok(())
    }

    #[test]
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
        let error = struct_scalar
            .to_arrow_datum()
            .err()
            .expect("struct scalar should error");
        assert!(error.to_string().contains("struct scalar conversion"));
    }

    #[test]
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

        let error = list_scalar
            .to_arrow_datum()
            .err()
            .expect("list scalar should error");
        assert!(error.to_string().contains("list scalar conversion"));
    }

    #[test]
    fn test_non_temporal_extension_to_arrow_todo() {
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        struct SomeExt;
        impl ExtVTable for SomeExt {
            type Metadata = String;
            type NativeValue<'a> = &'a str;

            #[expect(clippy::disallowed_methods, reason = "test-only id")]
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

        let error = scalar
            .to_arrow_datum()
            .err()
            .expect("non-temporal extension scalar should error");
        assert!(
            error
                .to_string()
                .contains("Cannot convert extension scalar")
        );
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

        let result = scalar.to_arrow_datum();
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

        let result = scalar.to_arrow_datum();
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

        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[rstest]
    #[case(TimeUnit::Nanoseconds, "UTC", 1234567890000000000i64)]
    #[case(TimeUnit::Microseconds, "America/New_York", 1234567890000000i64)]
    #[case(TimeUnit::Microseconds, "Asia/Qatar", 1234567890000000i64)]
    #[case(TimeUnit::Microseconds, "Australia/Sydney", 1234567890000000i64)]
    #[case(TimeUnit::Milliseconds, "Pacific/Honolulu", 1234567890000i64)]
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

        let result = scalar.to_arrow_datum();
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_with_null_value() {
        let scalar = Scalar::extension::<Time>(
            TimeUnit::Milliseconds,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        );

        let _result = scalar.to_arrow_datum().unwrap();
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
