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

use vortex::array::match_each_native_simd_ptype;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::Nullability::Nullable;
use vortex::dtype::PType;
use vortex::dtype::PType::I32;
use vortex::dtype::PType::I64;
use vortex::dtype::half::f16;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TemporalMetadata;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::extension::datetime::TimestampOptions;
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
            let lt = LogicalType::try_from(self.dtype())?;
            return Ok(Value::null(&lt));
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
            let lt = LogicalType::try_from(self.dtype())?;
            return Ok(Value::null(&lt));
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
        let logical_type = LogicalType::try_from(&DType::Extension(self.ext_dtype().clone()))?;
        let Some(temporal) = self.ext_dtype().metadata_opt::<AnyTemporal>() else {
            vortex_bail!("Cannot convert non-temporal extension scalar to duckdb value");
        };

        let value = || {
            self.to_storage_scalar()
                .as_primitive_opt()
                .ok_or_else(|| {
                    vortex_err!("Cannot have a temporal time type not packed by a primitive scalar")
                })?
                .as_::<i64>()
                .ok_or_else(|| vortex_err!("temporal types must be convertible to i64"))
        };

        Ok(match temporal {
            TemporalMetadata::Timestamp(unit, tz) => {
                if let Some(tz) = tz.as_ref() {
                    if tz.as_ref() != "UTC" {
                        // TODO(ngates): we should convert into UTC as DuckDB does internally.
                        //  I'm sure we can expose their timezone conversion functions to do this.
                        vortex_bail!(
                            "Currently only UTC timezone is supported for duckdb timestamp(tz) conversion"
                        );
                    }
                    return Ok(Value::new_timestamp_tz(value()?));
                }
                match unit {
                    TimeUnit::Nanoseconds => Value::new_timestamp_ns(value()?),
                    TimeUnit::Microseconds => Value::new_timestamp_us(value()?),
                    TimeUnit::Milliseconds => Value::new_timestamp_ms(value()?),
                    TimeUnit::Seconds => Value::new_timestamp_s(value()?),
                    TimeUnit::Days => {
                        vortex_bail!("timestamp(d) is cannot be converted to duckdb scalar")
                    }
                }
            }
            TemporalMetadata::Date(unit) => match unit {
                TimeUnit::Days => self
                    .to_storage_scalar()
                    .as_primitive_opt()
                    .ok_or_else(|| {
                        vortex_err!("temporal types must be backed by primitive scalars")
                    })?
                    .as_::<i32>()
                    .map(Value::new_date)
                    .unwrap_or_else(|| Value::null(&logical_type)),
                _ => vortex_bail!("cannot have TimeUnit {unit}, so represent a day"),
            },
            TemporalMetadata::Time(unit) => match unit {
                TimeUnit::Microseconds => Value::new_time(value()?),
                TimeUnit::Milliseconds => Value::new_time(value()? * 1000),
                TimeUnit::Seconds => Value::new_time(value()? * 1000 * 1000),
                TimeUnit::Nanoseconds | TimeUnit::Days => {
                    vortex_bail!("cannot convert timeunit {unit} to a duckdb MS time")
                }
            },
        })
    }
}

impl TryFrom<Value> for Scalar {
    type Error = VortexError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Scalar::try_from(&*value)
    }
}

impl<'a> TryFrom<&'a ValueRef> for Scalar {
    type Error = VortexError;

    fn try_from(value: &'a ValueRef) -> Result<Self, Self::Error> {
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
            ExtractedValue::UHugeInt(_) => {
                vortex_bail!("DuckDB UHugeInt is not yet supported in Vortex");
            }
            ExtractedValue::UTinyInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::USmallInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::UInteger(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::UBigInt(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Float(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Double(v) => Ok(Scalar::primitive(v, Nullable)),
            ExtractedValue::Varchar(s) => Ok(Scalar::utf8(s, Nullable)),
            ExtractedValue::Blob(b) => Ok(Scalar::binary(b, Nullable)),
            ExtractedValue::Date(days) => Ok(Scalar::extension::<Date>(
                TimeUnit::Days,
                Scalar::try_new(
                    DType::Primitive(I32, Nullable),
                    Some(ScalarValue::from(days)),
                )?,
            )),
            ExtractedValue::Time(micros) => Ok(Scalar::extension::<Time>(
                TimeUnit::Microseconds,
                Scalar::try_new(
                    DType::Primitive(I64, Nullable),
                    Some(ScalarValue::from(micros)),
                )?,
            )),
            ExtractedValue::TimestampNs(nanos) => Ok(Scalar::extension::<Timestamp>(
                TimestampOptions {
                    unit: TimeUnit::Nanoseconds,
                    tz: None,
                },
                Scalar::try_new(
                    DType::Primitive(I64, Nullable),
                    Some(ScalarValue::from(nanos)),
                )?,
            )),
            ExtractedValue::Timestamp(micros) => Ok(Scalar::extension::<Timestamp>(
                TimestampOptions {
                    unit: TimeUnit::Microseconds,
                    tz: None,
                },
                Scalar::try_new(
                    DType::Primitive(I64, Nullable),
                    Some(ScalarValue::from(micros)),
                )?,
            )),
            ExtractedValue::TimestampMs(millis) => Ok(Scalar::extension::<Timestamp>(
                TimestampOptions {
                    unit: TimeUnit::Milliseconds,
                    tz: None,
                },
                Scalar::try_new(
                    DType::Primitive(I64, Nullable),
                    Some(ScalarValue::from(millis)),
                )?,
            )),
            ExtractedValue::TimestampS(seconds) => Ok(Scalar::extension::<Timestamp>(
                TimestampOptions {
                    unit: TimeUnit::Seconds,
                    tz: None,
                },
                Scalar::try_new(
                    DType::Primitive(I64, Nullable),
                    Some(ScalarValue::from(seconds)),
                )?,
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
    use vortex::extension::datetime::Timestamp;
    use vortex::extension::datetime::TimestampOptions;
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
        use vortex::dtype::DType;
        use vortex::dtype::Nullability;
        use vortex::dtype::PType;
        use vortex::extension::datetime::TimeUnit;
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
            let original_scalar = Scalar::extension::<Timestamp>(
                TimestampOptions {
                    unit: time_unit,
                    tz: None,
                },
                Scalar::try_new(
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                    Some(ScalarValue::from(timestamp_value)),
                )
                .unwrap(),
            );

            let duckdb_value = original_scalar.try_to_duckdb_scalar().unwrap();
            let roundtrip_scalar: Scalar = duckdb_value.try_into().unwrap();

            assert_eq!(original_scalar, roundtrip_scalar);
        }
    }
}
