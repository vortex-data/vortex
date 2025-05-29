use duckdb::core::Value;
use duckdb::ffi::{duckdb_decimal, duckdb_hugeint};
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, PType, match_each_native_simd_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{
    BinaryScalar, BoolScalar, DecimalScalar, DecimalValue, ExtScalar, PrimitiveScalar, Scalar,
    Utf8Scalar,
};

pub trait ToDuckDBScalar {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value>;
}

impl ToDuckDBScalar for Scalar {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
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

        let decimal = duckdb_decimal {
            width: decimal_type.precision(),
            scale: decimal_type.scale().cast_unsigned(),
            value: duckdb_hugeint {
                // We want to truncate
                #[allow(clippy::cast_possible_truncation)]
                lower: huge_value as u64,
                upper: (huge_value >> 64) as i64,
            },
        };

        Ok(Value::from(decimal))
    }
}

impl ToDuckDBScalar for BoolScalar<'_> {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(Value::from(self.value()))
    }
}

impl ToDuckDBScalar for Utf8Scalar<'_> {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(match self.value() {
            Some(value) => Value::from(value.as_str()),
            None => Value::null(),
        })
    }
}

impl ToDuckDBScalar for BinaryScalar<'_> {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(match self.value() {
            Some(value) => Value::from(value.as_slice()),
            None => Value::null(),
        })
    }
}

impl ToDuckDBScalar for ExtScalar<'_> {
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
                TimeUnit::Us => Ok(Value::time_from_us(value()?)),
                TimeUnit::Ms => Ok(Value::time_from_us(value()? * 1000)),
                TimeUnit::S => Ok(Value::time_from_us(value()? * 1000 * 1000)),
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
                    .map(Value::date_from_day_count)
                    .unwrap_or_else(Value::null)),
                _ => vortex_bail!("cannot have TimeUnit {unit}, so represent a day"),
            },
            TemporalMetadata::Timestamp(unit, tz) => {
                if tz.is_some() {
                    todo!("timezones to duckdb scalar")
                }
                match unit {
                    TimeUnit::Ns => Ok(Value::timestamp_ns(value()?)),
                    TimeUnit::Us => Ok(Value::timestamp_us(value()?)),
                    TimeUnit::Ms => Ok(Value::timestamp_ms(value()?)),
                    TimeUnit::S => Ok(Value::timestamp_s(value()?)),
                    TimeUnit::D => {
                        vortex_bail!("timestamp(d) is cannot be converted to duckdb scalar")
                    }
                }
            }
        }
    }
}
