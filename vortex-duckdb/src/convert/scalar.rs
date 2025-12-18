// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar value conversion between Vortex and DuckDB.
//!
//! This module provides functionality to convert Vortex scalar values to DuckDB values.
//!
//! Note that nullability of Vortex scalars is not transferred to DuckDB scalars.
//!
//! # Supported Scalar Conversions
//!
//! | Vortex Scalar | DuckDB Value |
//! |---------------|--------------|
//! | `Null` | `NULL` |
//! | `Bool` | `BOOLEAN` |
//! | `Primitive` (integers/floats) | Corresponding numeric types |
//! | `Decimal` | `DECIMAL` |
//! | `Utf8` | `VARCHAR` |
//! | `Binary` | `BLOB` |
//! | `ExtScalar` (temporal) | `DATE`/`TIME`/`TIMESTAMP` |

use std::sync::Arc;

use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::ExtDType;
use vortex::dtype::Nullability::Nullable;
use vortex::dtype::PType;
use vortex::dtype::PType::I32;
use vortex::dtype::PType::I64;
use vortex::dtype::datetime::DATE_ID;
use vortex::dtype::datetime::TIME_ID;
use vortex::dtype::datetime::TIMESTAMP_ID;
use vortex::dtype::datetime::TemporalMetadata;
use vortex::dtype::datetime::TimeUnit;
use vortex::dtype::half::f16;
use vortex::dtype::match_each_native_simd_ptype;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::scalar::BinaryScalar;
use vortex::scalar::BoolScalar;
use vortex::scalar::DecimalScalar;
use vortex::scalar::DecimalValue;
use vortex::scalar::ExtScalar;
use vortex::scalar::PrimitiveScalar;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar::Utf8Scalar;

use crate::convert::dtype::FromLogicalType;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;
use crate::duckdb::ValueRef;

/// Trait for converting Vortex scalars to DuckDB values.
pub trait ToDuckDBScalar {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value>;
}

impl ToDuckDBScalar for Scalar {
    /// Converts a generic Vortex scalar to a DuckDB value.
    ///
    /// # Note
    ///
    /// Struct and List scalars are not yet implemented and cause a panic.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        if self.is_null() {
            return Ok(Value::null(&LogicalType::try_from(self.dtype())?));
        }

        match self.dtype() {
            DType::Null => Ok(Value::sql_null()),
            DType::Bool(_) => self.as_bool().try_to_duckdb_scalar(),
            DType::Primitive(..) => self.as_primitive().try_to_duckdb_scalar(),
            DType::Decimal(..) => self.as_decimal().try_to_duckdb_scalar(),
            DType::Extension(..) => self.as_extension().try_to_duckdb_scalar(),
            DType::Utf8(_) => self.as_utf8().try_to_duckdb_scalar(),
            DType::Binary(_) => self.as_binary().try_to_duckdb_scalar(),
            DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) => todo!(),
        }
    }
}

impl ToDuckDBScalar for PrimitiveScalar<'_> {
    /// Converts a primitive scalar (integer, float, or boolean) to a DuckDB value.
    ///
    /// # Note
    ///
    /// - `F16` values are converted to `F32` before creating the DuckDB value
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        if self.ptype() == PType::F16 {
            return Value::try_from(self.as_::<f16>().map(|f| f.to_f32()));
        }
        match_each_native_simd_ptype!(self.ptype(), |P| { Ok(Value::try_from(self.as_::<P>())?) })
    }
}

impl ToDuckDBScalar for DecimalScalar<'_> {
    /// Converts a decimal scalar to a DuckDB decimal value.
    ///
    /// # Supported Decimal Types
    ///
    /// - `I8`, `I16`, `I32`, `I64` - Converted to `i128` for DuckDB
    /// - `I128` - Used directly
    /// - `I256` - Not supported, returns an error
    ///
    /// # Note: Scalar vs Array Conversion Differences
    ///
    /// This scalar conversion always uses `i128` for all decimal values regardless of precision,
    /// which differs from the array conversion logic that uses precision-based storage optimization.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        let decimal_type = self
            .dtype()
            .as_decimal_opt()
            .ok_or_else(|| vortex_err!("decimal scalar without decimal dtype"))?;

        let Some(decimal_value) = self.decimal_value() else {
            return Ok(Value::null(&LogicalType::try_from(self.dtype())?));
        };

        let huge_value = match decimal_value {
            DecimalValue::I8(v) => v as i128,
            DecimalValue::I16(v) => v as i128,
            DecimalValue::I32(v) => v as i128,
            DecimalValue::I64(v) => v as i128,
            DecimalValue::I128(v) => v,
            DecimalValue::I256(_) => vortex_bail!("cannot handle a i256 decimal in duckdb"),
        };

        Ok(Value::new_decimal(
            decimal_type.precision(),
            decimal_type.scale(),
            huge_value,
        ))
    }
}

impl ToDuckDBScalar for BoolScalar<'_> {
    /// Converts a boolean scalar to a DuckDB boolean value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Value::try_from(self.value())
    }
}

impl ToDuckDBScalar for Utf8Scalar<'_> {
    /// Converts a UTF-8 string scalar to a DuckDB VARCHAR value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(match self.value() {
            Some(value) => Value::from(value.as_str()),
            None => Value::null(&LogicalType::varchar()),
        })
    }
}

impl ToDuckDBScalar for BinaryScalar<'_> {
    /// Converts a binary scalar to a DuckDB BLOB value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(match self.value() {
            Some(value) => Value::from(value.as_slice()),
            None => Value::null(&LogicalType::blob()),
        })
    }
}

impl ToDuckDBScalar for ExtScalar<'_> {
    /// Converts an extension scalar (primarily temporal types) to a DuckDB value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        let logical_type =
            LogicalType::try_from(&DType::Extension(Arc::new(self.ext_dtype().clone())))?;
        let time = TemporalMetadata::try_from(self.ext_dtype())?;
        let value = || {
            self.storage()
                .as_primitive_opt()
                .ok_or_else(|| {
                    vortex_err!("Cannot have a temporal time type not packed by a primitive scalar")
                })?
                .as_::<i64>()
                .ok_or_else(|| vortex_err!("temporal types must be convertible to i64"))
        };
        match time {
            TemporalMetadata::Time(unit) => match unit {
                TimeUnit::Microseconds => Ok(Value::new_time(value()?)),
                TimeUnit::Milliseconds => Ok(Value::new_time(value()? * 1000)),
                TimeUnit::Seconds => Ok(Value::new_time(value()? * 1000 * 1000)),
                TimeUnit::Nanoseconds | TimeUnit::Days => {
                    vortex_bail!("cannot convert timeunit {unit} to a duckdb MS time")
                }
            },
            TemporalMetadata::Date(unit) => match unit {
                TimeUnit::Days => Ok(self
                    .storage()
                    .as_primitive_opt()
                    .ok_or_else(|| {
                        vortex_err!("temporal types must be backed by primitive scalars")
                    })?
                    .as_::<i32>()
                    .map(Value::new_date)
                    .unwrap_or_else(|| Value::null(&logical_type))),
                _ => vortex_bail!("cannot have TimeUnit {unit}, so represent a day"),
            },
            TemporalMetadata::Timestamp(unit, tz) => {
                if let Some(tz) = tz {
                    if tz != "UTC" {
                        todo!()
                    }
                    return Ok(Value::new_timestamp_tz(value()?));
                }
                match unit {
                    TimeUnit::Nanoseconds => Ok(Value::new_timestamp_ns(value()?)),
                    TimeUnit::Microseconds => Ok(Value::new_timestamp_us(value()?)),
                    TimeUnit::Milliseconds => Ok(Value::new_timestamp_ms(value()?)),
                    TimeUnit::Seconds => Ok(Value::new_timestamp_s(value()?)),
                    TimeUnit::Days => {
                        vortex_bail!("timestamp(d) is cannot be converted to duckdb scalar")
                    }
                }
            }
        }
    }
}

impl TryFrom<Value> for Scalar {
    type Error = VortexError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Scalar::try_from(value.as_ref())
    }
}

impl<'a> TryFrom<ValueRef<'a>> for Scalar {
    type Error = VortexError;

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        use crate::duckdb::ExtractedValue;
        let dtype = DType::from_logical_type(value.logical_type(), Nullable)?;
        match value.extract() {
            ExtractedValue::Null => Ok(Scalar::null(dtype)),
            ExtractedValue::Boolean(b) => Ok(Scalar::bool(b, Nullable)),
            ExtractedValue::TinyInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::SmallInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Integer(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::BigInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::HugeInt(_) => {
                vortex_bail!("DuckDB HugeInt is not yet supported in Vortex");
            }
            ExtractedValue::UTinyInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::USmallInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::UInteger(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::UBigInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Float(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Double(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Varchar(s) => Ok(Scalar::utf8(s, Nullable)),
            ExtractedValue::Blob(b) => Ok(Scalar::binary(b, Nullable)),
            ExtractedValue::Date(days) => Ok(Scalar::extension(
                Arc::new(ExtDType::new(
                    DATE_ID.clone(),
                    Arc::new(DType::Primitive(I32, Nullable)),
                    Some(TemporalMetadata::Date(TimeUnit::Days).into()),
                )),
                Scalar::new(DType::Primitive(I32, Nullable), ScalarValue::from(days)),
            )),
            ExtractedValue::Time(micros) => Ok(Scalar::extension(
                Arc::new(ExtDType::new(
                    TIME_ID.clone(),
                    Arc::new(DType::Primitive(I64, Nullable)),
                    Some(TemporalMetadata::Time(TimeUnit::Microseconds).into()),
                )),
                Scalar::new(DType::Primitive(I64, Nullable), ScalarValue::from(micros)),
            )),
            ExtractedValue::TimestampNs(nanos) => Ok(Scalar::extension(
                Arc::new(ExtDType::new(
                    TIMESTAMP_ID.clone(),
                    Arc::new(DType::Primitive(I64, Nullable)),
                    Some(TemporalMetadata::Timestamp(TimeUnit::Nanoseconds, None).into()),
                )),
                Scalar::new(DType::Primitive(I64, Nullable), ScalarValue::from(nanos)),
            )),
            ExtractedValue::Timestamp(micros) => Ok(Scalar::extension(
                Arc::new(ExtDType::new(
                    TIMESTAMP_ID.clone(),
                    Arc::new(DType::Primitive(I64, Nullable)),
                    Some(TemporalMetadata::Timestamp(TimeUnit::Microseconds, None).into()),
                )),
                Scalar::new(DType::Primitive(I64, Nullable), ScalarValue::from(micros)),
            )),
            ExtractedValue::TimestampMs(millis) => Ok(Scalar::extension(
                Arc::new(ExtDType::new(
                    TIMESTAMP_ID.clone(),
                    Arc::new(DType::Primitive(I64, Nullable)),
                    Some(TemporalMetadata::Timestamp(TimeUnit::Milliseconds, None).into()),
                )),
                Scalar::new(DType::Primitive(I64, Nullable), ScalarValue::from(millis)),
            )),
            ExtractedValue::TimestampS(seconds) => Ok(Scalar::extension(
                Arc::new(ExtDType::new(
                    TIMESTAMP_ID.clone(),
                    Arc::new(DType::Primitive(I64, Nullable)),
                    Some(TemporalMetadata::Timestamp(TimeUnit::Seconds, None).into()),
                )),
                Scalar::new(DType::Primitive(I64, Nullable), ScalarValue::from(seconds)),
            )),
            ExtractedValue::Decimal(precision, scale, value) => Ok(Scalar::decimal(
                DecimalValue::I128(value),
                DecimalDType::try_new(precision, scale)?,
                Nullable,
            )),
            ExtractedValue::List(vs) => match dtype {
                DType::List(c, _) => Ok(Scalar::list(
                    c,
                    vs.into_iter()
                        .map(Scalar::try_from)
                        .collect::<VortexResult<Vec<_>>>()?,
                    Nullable,
                )),
                DType::Struct(..) => Ok(Scalar::struct_(
                    dtype,
                    vs.into_iter()
                        .map(Scalar::try_from)
                        .collect::<VortexResult<Vec<_>>>()?,
                )),
                _ => {
                    vortex_bail!("List value must be a list or struct dtype")
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::scalar::Scalar;

    use crate::convert::ToDuckDBScalar;

    #[test]
    fn test_scalar_round_trip() {
        let value = Scalar::from(32i32);
        assert_eq!(
            value,
            value.try_to_duckdb_scalar().unwrap().try_into().unwrap()
        );

        let value = Scalar::from("hello");
        assert_eq!(
            value,
            value.try_to_duckdb_scalar().unwrap().try_into().unwrap()
        );

        let value = Scalar::from(1.0f64);
        assert_eq!(
            value,
            value.try_to_duckdb_scalar().unwrap().try_into().unwrap()
        );
    }

    #[test]
    fn test_timestamp_roundtrip() {
        use std::sync::Arc;

        use vortex::dtype::DType;
        use vortex::dtype::ExtDType;
        use vortex::dtype::Nullability;
        use vortex::dtype::PType;
        use vortex::dtype::datetime::TIMESTAMP_ID;
        use vortex::dtype::datetime::TemporalMetadata;
        use vortex::dtype::datetime::TimeUnit;
        use vortex::scalar::Scalar;
        use vortex::scalar::ScalarValue;

        #[rustfmt::skip]
        let test_cases = [
            (TimeUnit::Seconds, 1703980800i64),                 // 2023-12-30 16:00:00 UTC
            (TimeUnit::Milliseconds, 1703980800123i64),         // 2023-12-30 16:00:00.123 UTC
            (TimeUnit::Microseconds, 1703980800123456i64),      // 2023-12-30 16:00:00.123456 UTC
            (TimeUnit::Nanoseconds, 1703980800123456789i64),    // 2023-12-30 16:00:00.123456789 UTC
        ];

        for (time_unit, timestamp_value) in test_cases {
            let ext_dtype = Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                Some(TemporalMetadata::Timestamp(time_unit, None).into()),
            ));

            let original_scalar = Scalar::extension(
                ext_dtype,
                Scalar::new(
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                    ScalarValue::from(timestamp_value),
                ),
            );

            let duckdb_value = original_scalar.try_to_duckdb_scalar().unwrap();
            let roundtrip_scalar: Scalar = duckdb_value.try_into().unwrap();

            assert_eq!(original_scalar, roundtrip_scalar);
        }
    }
}
