use std::sync::Arc;

use duckdb::core::{LogicalTypeHandle, LogicalTypeId};
use vortex_dtype::Nullability::Nullable;
use vortex_dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, ExtDType, Nullability, PType, StructDType};
use vortex_error::{VortexResult, vortex_bail};

pub trait FromDuckDBType<A> {
    // Nullable is inferred from the `NotNullConstraint`.
    fn from_duckdb(array: A, nullable: Nullability) -> VortexResult<Self>
    where
        Self: Sized;
}

impl FromDuckDBType<LogicalTypeHandle> for DType {
    // Converts a DuckDB logical type handle to a `DType` based on the logical type ID.
    fn from_duckdb(type_: LogicalTypeHandle, nullable: Nullability) -> VortexResult<Self> {
        match type_.id() {
            LogicalTypeId::SQLNull => Ok(DType::Null),
            LogicalTypeId::Boolean => Ok(DType::Bool(nullable)),
            LogicalTypeId::Tinyint => Ok(DType::Primitive(PType::I8, nullable)),
            LogicalTypeId::Smallint => Ok(DType::Primitive(PType::I16, nullable)),
            LogicalTypeId::Integer => Ok(DType::Primitive(PType::I32, nullable)),
            LogicalTypeId::Bigint => Ok(DType::Primitive(PType::I64, nullable)),
            LogicalTypeId::UTinyint => Ok(DType::Primitive(PType::U8, nullable)),
            LogicalTypeId::USmallint => Ok(DType::Primitive(PType::U16, nullable)),
            LogicalTypeId::UInteger => Ok(DType::Primitive(PType::U32, nullable)),
            LogicalTypeId::UBigint => Ok(DType::Primitive(PType::U64, nullable)),
            LogicalTypeId::Float => Ok(DType::Primitive(PType::F32, nullable)),
            LogicalTypeId::Double => Ok(DType::Primitive(PType::F64, nullable)),
            LogicalTypeId::Varchar => Ok(DType::Utf8(nullable)),
            LogicalTypeId::Blob => Ok(DType::Binary(nullable)),
            LogicalTypeId::Struct => Ok(DType::Struct(Arc::new(from_duckdb_struct(type_)?), nullable)),
            LogicalTypeId::List => Ok(DType::List(Arc::new(from_duckdb_list(type_)?), nullable)),
            LogicalTypeId::Date => Ok(DType::Extension(
                Arc::new(ExtDType::new(DATE_ID.clone(),
                                       Arc::new(DType::Primitive(PType::I32, nullable)),
                                       Some(TemporalMetadata::Date(TimeUnit::D).into())))
            )),
            LogicalTypeId::Time => Ok(DType::Extension(
                Arc::new(ExtDType::new(TIME_ID.clone(),
                                       Arc::new(DType::Primitive(PType::I32, nullable)),
                                       Some(TemporalMetadata::Time(TimeUnit::Us).into())))
            )),
                LogicalTypeId::Timestamp
                | LogicalTypeId::TimestampS
                | LogicalTypeId::TimestampMs
                | LogicalTypeId::TimestampNs
                => Ok(DType::Extension(
                    Arc::new(ExtDType::new(TIMESTAMP_ID.clone(),
                                           Arc::new(DType::Primitive(PType::I64, nullable)),
                                           Some(TemporalMetadata::Timestamp(timestamp_time_unit(type_.id())?, None).into())))
                )),
            | LogicalTypeId::Interval
            // Hugeint is a i128
            | LogicalTypeId::Hugeint
            | LogicalTypeId::Decimal
            | LogicalTypeId::Enum
            | LogicalTypeId::Map
            | LogicalTypeId::Uuid
            | LogicalTypeId::Union
            | LogicalTypeId::TimestampTZ => vortex_bail!("missing type: {:?}", type_),
            LogicalTypeId::Invalid => vortex_bail!("cannot handle invalid type")
        }
    }
}

fn timestamp_time_unit(type_id: LogicalTypeId) -> VortexResult<TimeUnit> {
    match type_id {
        LogicalTypeId::TimestampS => Ok(TimeUnit::S),
        LogicalTypeId::TimestampMs => Ok(TimeUnit::Ms),
        LogicalTypeId::Timestamp => Ok(TimeUnit::Us),
        LogicalTypeId::TimestampNs => Ok(TimeUnit::Ns),
        _ => vortex_bail!("invalid type_id for function"),
    }
}

fn from_duckdb_list(list: LogicalTypeHandle) -> VortexResult<DType> {
    // Note: the zeroth child of a list is the element type
    assert_eq!(list.num_children(), 1);
    // TODO: is there list element nullability
    FromDuckDBType::from_duckdb(list.child(0), Nullable)
}

fn from_duckdb_struct(struct_: LogicalTypeHandle) -> VortexResult<StructDType> {
    (0..struct_.num_children())
        .map(|i| {
            // TODO: is there struct field nullability
            let child_nullability = Nullable;
            let child_name = struct_.child_name(i);
            let child_type = DType::from_duckdb(struct_.child(i), child_nullability)?;
            Ok((child_name, child_type))
        })
        .collect()
}
