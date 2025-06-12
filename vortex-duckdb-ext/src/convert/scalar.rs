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

use vortex::dtype::datetime::{TemporalMetadata, TimeUnit};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, PType, match_each_native_simd_ptype};
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::scalar::{
    BinaryScalar, BoolScalar, DecimalScalar, DecimalValue, ExtScalar, PrimitiveScalar, Scalar,
    Utf8Scalar,
};

use crate::duckdb::Value;

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
            return Ok(Value::null());
        }

        match self.dtype() {
            DType::Null => Ok(Value::null()),
            DType::Bool(_) => self.as_bool().try_to_duckdb_scalar(),
            DType::Primitive(..) => self.as_primitive().try_to_duckdb_scalar(),
            DType::Decimal(..) => self.as_decimal().try_to_duckdb_scalar(),
            DType::Extension(..) => self.as_extension().try_to_duckdb_scalar(),
            DType::Utf8(_) => self.as_utf8().try_to_duckdb_scalar(),
            DType::Binary(_) => self.as_binary().try_to_duckdb_scalar(),
            DType::Struct(..) | DType::List(..) => todo!(),
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
            return Ok(Value::from(
                self.as_::<f16>()
                    .vortex_expect("check ptyped")
                    .map(|f| f.to_f32()),
            ));
        }
        match_each_native_simd_ptype!(self.ptype(), |P| {
            Ok(Value::from(
                self.as_::<P>().vortex_expect("ptype value mismatch"),
            ))
        })
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
            .as_decimal()
            .ok_or_else(|| vortex_err!("decimal scalar without decimal dtype"))?;

        let Some(decimal_value) = self.decimal_value() else {
            return Ok(Value::null());
        };

        let huge_value = match decimal_value {
            DecimalValue::I8(v) => *v as i128,
            DecimalValue::I16(v) => *v as i128,
            DecimalValue::I32(v) => *v as i128,
            DecimalValue::I64(v) => *v as i128,
            DecimalValue::I128(v) => *v,
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
        Ok(Value::from(self.value()))
    }
}

impl ToDuckDBScalar for Utf8Scalar<'_> {
    /// Converts a UTF-8 string scalar to a DuckDB VARCHAR value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(match self.value() {
            Some(value) => Value::from(value.as_str()),
            None => Value::null(),
        })
    }
}

impl ToDuckDBScalar for BinaryScalar<'_> {
    /// Converts a binary scalar to a DuckDB BLOB value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(match self.value() {
            Some(value) => Value::from(value.as_slice()),
            None => Value::null(),
        })
    }
}

impl ToDuckDBScalar for ExtScalar<'_> {
    /// Converts an extension scalar (primarily temporal types) to a DuckDB value.
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        let time = TemporalMetadata::try_from(self.ext_dtype())?;
        let value = || {
            self.storage()
                .as_primitive_opt()
                .ok_or_else(|| {
                    vortex_err!("Cannot have a temporal time type not packed by a primitive scalar")
                })?
                .as_::<i64>()?
                .ok_or_else(|| vortex_err!("temporal types must be convertable to i64"))
        };
        match time {
            TemporalMetadata::Time(unit) => match unit {
                TimeUnit::Us => Ok(Value::new_time(value()?)),
                TimeUnit::Ms => Ok(Value::new_time(value()? * 1000)),
                TimeUnit::S => Ok(Value::new_time(value()? * 1000 * 1000)),
                TimeUnit::Ns | TimeUnit::D => {
                    vortex_bail!("cannot convert timeunit {unit} to a duckdb MS time")
                }
            },
            TemporalMetadata::Date(unit) => match unit {
                TimeUnit::D => Ok(self
                    .storage()
                    .as_primitive_opt()
                    .ok_or_else(|| {
                        vortex_err!("temporal types must be backed by primitive scalars")
                    })?
                    .as_::<i32>()?
                    .map(Value::new_date)
                    .unwrap_or_else(Value::null)),
                _ => vortex_bail!("cannot have TimeUnit {unit}, so represent a day"),
            },
            TemporalMetadata::Timestamp(unit, tz) => {
                if tz.is_some() {
                    todo!("timezones to duckdb scalar")
                }
                match unit {
                    TimeUnit::Ns => Ok(Value::new_timestamp_ns(value()?)),
                    TimeUnit::Us => Ok(Value::new_timestamp_us(value()?)),
                    TimeUnit::Ms => Ok(Value::new_timestamp_ms(value()?)),
                    TimeUnit::S => Ok(Value::new_timestamp_s(value()?)),
                    TimeUnit::D => {
                        vortex_bail!("timestamp(d) is cannot be converted to duckdb scalar")
                    }
                }
            }
        }
    }
}
