use duckdb::core::{LogicalTypeHandle, LogicalTypeId};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

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
            DType::List(..) | DType::Extension(_) => todo!(),
        }
    }
}
