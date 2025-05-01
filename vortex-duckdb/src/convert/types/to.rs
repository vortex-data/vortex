use duckdb::core::{LogicalTypeHandle, LogicalTypeId};
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit, is_temporal_ext_type};
use vortex_dtype::{DType, ExtDType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

pub trait ToDuckDBType {
    fn to_duckdb_type(&self) -> VortexResult<LogicalTypeHandle>;
}

impl ToDuckDBType for DType {
    fn to_duckdb_type(&self) -> VortexResult<LogicalTypeHandle> {
        // TODO(joe): handle nullability.
        match self {
            DType::Null => vortex_bail!("cannot convert null to duckdb type"),
            DType::Bool(_) => Ok(LogicalTypeHandle::from(LogicalTypeId::Boolean)),
            DType::Primitive(ptype, _) => Ok(LogicalTypeHandle::from(match ptype {
                PType::I8 => LogicalTypeId::Tinyint,
                PType::I16 => LogicalTypeId::Smallint,
                PType::I32 => LogicalTypeId::Integer,
                PType::I64 => LogicalTypeId::Bigint,
                PType::U8 => LogicalTypeId::UTinyint,
                PType::U16 => LogicalTypeId::USmallint,
                PType::U32 => LogicalTypeId::UInteger,
                PType::U64 => LogicalTypeId::UBigint,
                PType::F32 => LogicalTypeId::Float,
                PType::F64 => LogicalTypeId::Double,
                PType::F16 => vortex_bail!("cannot convert f16 to duckdb type"),
            })),
            DType::Decimal(decimal_type, _) => Ok(LogicalTypeHandle::decimal(
                decimal_type.precision(),
                decimal_type.scale().try_into()?,
            )),
            DType::Utf8(_) => Ok(LogicalTypeHandle::from(LogicalTypeId::Varchar)),
            DType::Binary(_) => Ok(LogicalTypeHandle::from(LogicalTypeId::Blob)),
            DType::Struct(struct_, _) => {
                let duckdb_type = LogicalTypeHandle::struct_type(
                    struct_
                        .names()
                        .iter()
                        .zip(struct_.fields())
                        .map(|(name, field)| Ok((name.as_ref(), field.to_duckdb_type()?)))
                        .collect::<VortexResult<Vec<_>>>()?
                        .as_slice(),
                );
                Ok(duckdb_type)
            }
            DType::Extension(ext_dtype) => {
                if is_temporal_ext_type(ext_dtype.id()) {
                    Ok(ext_to_duckdb(ext_dtype))
                } else {
                    vortex_bail!("Unsupported extension type \"{}\"", ext_dtype.id())
                }
            }
            DType::List(..) => todo!("type: {self:?}"),
        }
    }
}

/// Convert temporal ExtDType to a corresponding LogicalType
///
/// panics if the ext_dtype is not a temporal dtype
pub fn ext_to_duckdb(ext_dtype: &ExtDType) -> LogicalTypeHandle {
    match TemporalMetadata::try_from(ext_dtype)
        .vortex_expect("make_arrow_temporal_dtype must be called with a temporal ExtDType")
    {
        TemporalMetadata::Date(time_unit) => match time_unit {
            TimeUnit::D => LogicalTypeHandle::from(LogicalTypeId::Date),
            _ => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", time_unit, ext_dtype.id())
            }
        },
        TemporalMetadata::Time(time_unit) => match time_unit {
            TimeUnit::Us => LogicalTypeHandle::from(LogicalTypeId::Time),
            _ => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", time_unit, ext_dtype.id())
            }
        },
        TemporalMetadata::Timestamp(time_unit, tz) => {
            if tz.is_some() {
                vortex_panic!(InvalidArgument: "Timestamp with timezone is not yet supported")
            }
            match time_unit {
                TimeUnit::Ns => LogicalTypeHandle::from(LogicalTypeId::TimestampNs),
                TimeUnit::Us => LogicalTypeHandle::from(LogicalTypeId::Timestamp),
                TimeUnit::Ms => LogicalTypeHandle::from(LogicalTypeId::TimestampMs),
                TimeUnit::S => LogicalTypeHandle::from(LogicalTypeId::TimestampS),
                _ => {
                    vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", time_unit, ext_dtype.id())
                }
            }
        }
    }
}
