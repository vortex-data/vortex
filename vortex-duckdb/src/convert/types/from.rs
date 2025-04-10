use std::sync::Arc;

use duckdb::core::{LogicalTypeHandle, LogicalTypeId};
use vortex_dtype::Nullability::Nullable;
use vortex_dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, ExtDType, Nullability, PType, StructDType};

pub trait FromDuckDBType<A> {
    // Nullable is inferred from the `NotNullConstraint`.
    fn from_duckdb(array: A, nullable: Nullability) -> Self;
}

impl FromDuckDBType<LogicalTypeHandle> for DType {
    // Converts a DuckDB logical type handle to a `DType` based on the logical type ID.
    fn from_duckdb(type_: LogicalTypeHandle, nullable: Nullability) -> Self {
        match type_.id() {
            LogicalTypeId::SQLNull => DType::Null,
            LogicalTypeId::Boolean => DType::Bool(nullable),
            LogicalTypeId::Tinyint => DType::Primitive(PType::I8, nullable),
            LogicalTypeId::Smallint => DType::Primitive(PType::I16, nullable),
            LogicalTypeId::Integer => DType::Primitive(PType::I32, nullable),
            LogicalTypeId::Bigint => DType::Primitive(PType::I64, nullable),
            LogicalTypeId::UTinyint => DType::Primitive(PType::U8, nullable),
            LogicalTypeId::USmallint => DType::Primitive(PType::U16, nullable),
            LogicalTypeId::UInteger => DType::Primitive(PType::U32, nullable),
            LogicalTypeId::UBigint => DType::Primitive(PType::U64, nullable),
            LogicalTypeId::Float => DType::Primitive(PType::F32, nullable),
            LogicalTypeId::Double => DType::Primitive(PType::F64, nullable),
            LogicalTypeId::Varchar => DType::Utf8(nullable),
            LogicalTypeId::Blob => DType::Binary(nullable),
            LogicalTypeId::Struct => DType::Struct(Arc::new(from_duckdb_struct(type_)), nullable),
            LogicalTypeId::List => DType::List(Arc::new(from_duckdb_list(type_)), nullable),
            LogicalTypeId::Date => DType::Extension(
                Arc::new(ExtDType::new(DATE_ID.clone(),
                                       Arc::new(DType::Primitive(PType::I32, nullable)),
                                       Some(TemporalMetadata::Date(TimeUnit::D).into())))
            ),
            LogicalTypeId::Time => DType::Extension(
                Arc::new(ExtDType::new(TIME_ID.clone(),
                                       Arc::new(DType::Primitive(PType::I32, nullable)),
                                       Some(TemporalMetadata::Time(TimeUnit::Us).into())))
            ),
                LogicalTypeId::Timestamp
                | LogicalTypeId::TimestampS
                | LogicalTypeId::TimestampMs
                | LogicalTypeId::TimestampNs
                => DType::Extension(
                    Arc::new(ExtDType::new(TIMESTAMP_ID.clone(),
                                           Arc::new(DType::Primitive(PType::I64, nullable)),
                                           Some(TemporalMetadata::Timestamp(timestamp_time_unit(type_.id()), None).into())))
                ),
            | LogicalTypeId::Interval
            // Hugeint is a i128
            | LogicalTypeId::Hugeint
            | LogicalTypeId::Decimal
            | LogicalTypeId::Enum
            | LogicalTypeId::Map
            | LogicalTypeId::Uuid
            | LogicalTypeId::Union
            | LogicalTypeId::TimestampTZ => todo!("missing type: {:?}", type_),
        }
    }
}

fn timestamp_time_unit(type_id: LogicalTypeId) -> TimeUnit {
    match type_id {
        LogicalTypeId::TimestampS => TimeUnit::S,
        LogicalTypeId::TimestampMs => TimeUnit::Ms,
        LogicalTypeId::Timestamp => TimeUnit::Us,
        LogicalTypeId::TimestampNs => TimeUnit::Ns,
        _ => unreachable!("invalid type_id for function"),
    }
}

fn from_duckdb_list(list: LogicalTypeHandle) -> DType {
    // Note: the zeroth child of a list is the element type
    assert_eq!(list.num_children(), 1);
    // TODO: is there list element nullability
    FromDuckDBType::from_duckdb(list.child(0), Nullable)
}

fn from_duckdb_struct(struct_: LogicalTypeHandle) -> StructDType {
    (0..struct_.num_children())
        .map(|i| {
            // TODO: is there struct field nullability
            let child_nullability = Nullable;
            let child_name = struct_.child_name(i);
            let child_type = DType::from_duckdb(struct_.child(i), child_nullability);
            (child_name, child_type)
        })
        .collect()
}
