//! Convert between Vortex [vortex_dtype::DType] and Apache Arrow [arrow_schema::DataType].
//!
//! Apache Arrow's type system includes physical information, which could lead to ambiguities as
//! Vortex treats encodings as separate from logical types.
//!
//! [`infer_schema`] and its sibling [`infer_data_type`] use a simple algorithm, where every
//! logical type is encoded in its simplest corresponding Arrow type. This reflects the reality that
//! most compute engines don't make use of the entire type range arrow-rs supports.
//!
//! For this reason, it's recommended to do as much computation as possible within Vortex, and then
//! materialize an Arrow ArrayRef at the very end of the processing chain.

use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaBuilder};
use vortex_datetime_dtype::arrow::make_arrow_temporal_dtype;
use vortex_datetime_dtype::is_temporal_ext_type;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::vortex_panic;

/// Convert a Vortex [struct DType][DType] to an Arrow [Schema].
///
/// # Panics
///
/// This function will panic if the provided `dtype` is not a StructDType, or if the struct DType
/// has top-level nullability.
pub(crate) fn infer_schema(dtype: &DType) -> Schema {
    let DType::Struct(struct_dtype, nullable) = dtype else {
        vortex_panic!("only DType::Struct can be converted to arrow schema");
    };

    if *nullable != Nullability::NonNullable {
        vortex_panic!("top-level struct in Schema must be NonNullable");
    }

    let mut builder = SchemaBuilder::with_capacity(struct_dtype.names().len());
    for (field_name, field_dtype) in struct_dtype
        .names()
        .iter()
        .zip(struct_dtype.dtypes().iter())
    {
        builder.push(FieldRef::from(Field::new(
            field_name.to_string(),
            infer_data_type(field_dtype),
            field_dtype.is_nullable(),
        )));
    }

    builder.finish()
}

pub(crate) fn infer_data_type(dtype: &DType) -> DataType {
    match dtype {
        DType::Null => DataType::Null,
        DType::Bool(_) => DataType::Boolean,
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 => DataType::UInt8,
            PType::U16 => DataType::UInt16,
            PType::U32 => DataType::UInt32,
            PType::U64 => DataType::UInt64,
            PType::I8 => DataType::Int8,
            PType::I16 => DataType::Int16,
            PType::I32 => DataType::Int32,
            PType::I64 => DataType::Int64,
            PType::F16 => DataType::Float16,
            PType::F32 => DataType::Float32,
            PType::F64 => DataType::Float64,
        },
        DType::Utf8(_) => DataType::Utf8,
        DType::Binary(_) => DataType::Binary,
        DType::Struct(struct_dtype, _) => {
            let mut fields = Vec::with_capacity(struct_dtype.names().len());
            for (field_name, field_dt) in struct_dtype
                .names()
                .iter()
                .zip(struct_dtype.dtypes().iter())
            {
                fields.push(FieldRef::from(Field::new(
                    field_name.to_string(),
                    infer_data_type(field_dt),
                    field_dt.is_nullable(),
                )));
            }

            DataType::Struct(Fields::from(fields))
        }
        DType::List(list_dt, _) => {
            let dtype: &DType = list_dt;
            DataType::List(FieldRef::from(Field::new(
                "element",
                infer_data_type(dtype),
                dtype.is_nullable(),
            )))
        }
        DType::Extension(ext_dtype) => {
            // Special case: the Vortex logical type system represents many temporal types from
            // Arrow, and we want those to serialize properly.
            if is_temporal_ext_type(ext_dtype.id()) {
                make_arrow_temporal_dtype(ext_dtype)
            } else {
                // All other extension types, we rely on the scalar type to determine how it gets
                // pushed to Arrow.
                infer_data_type(ext_dtype.storage_dtype())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability, PType, StructDType};

    use super::*;

    #[test]
    fn test_dtype_conversion_success() {
        assert_eq!(infer_data_type(&DType::Null), DataType::Null);

        assert_eq!(
            infer_data_type(&DType::Bool(Nullability::NonNullable)),
            DataType::Boolean
        );

        assert_eq!(
            infer_data_type(&DType::Primitive(PType::U64, Nullability::NonNullable)),
            DataType::UInt64
        );

        assert_eq!(
            infer_data_type(&DType::Utf8(Nullability::NonNullable)),
            DataType::Utf8
        );

        assert_eq!(
            infer_data_type(&DType::Binary(Nullability::NonNullable)),
            DataType::Binary
        );

        assert_eq!(
            infer_data_type(&DType::List(
                Arc::new(DType::Bool(Nullability::NonNullable)),
                Nullability::Nullable,
            )),
            DataType::List(FieldRef::from(Field::new(
                "element".to_string(),
                DataType::Boolean,
                false,
            )))
        );

        assert_eq!(
            infer_data_type(&DType::Struct(
                StructDType::new(
                    FieldNames::from(vec![FieldName::from("field_a"), FieldName::from("field_b")]),
                    vec![DType::Bool(false.into()), DType::Utf8(true.into())],
                ),
                Nullability::NonNullable,
            )),
            DataType::Struct(Fields::from(vec![
                FieldRef::from(Field::new("field_a", DataType::Boolean, false)),
                FieldRef::from(Field::new("field_b", DataType::Utf8, true)),
            ]))
        );
    }

    #[test]
    fn test_schema_conversion() {
        let struct_dtype = the_struct();
        let schema_nonnull = DType::Struct(struct_dtype.clone(), Nullability::NonNullable);

        assert_eq!(
            infer_schema(&schema_nonnull),
            Schema::new(Fields::from(vec![
                Field::new("field_a", DataType::Boolean, false),
                Field::new("field_b", DataType::Utf8, false),
                Field::new("field_c", DataType::Int32, true),
            ]))
        );
    }

    #[test]
    #[should_panic]
    fn test_schema_conversion_panics() {
        let struct_dtype = the_struct();
        let schema_null = DType::Struct(struct_dtype.clone(), Nullability::Nullable);
        let _ = infer_schema(&schema_null);
    }

    fn the_struct() -> StructDType {
        StructDType::new(
            FieldNames::from([
                FieldName::from("field_a"),
                FieldName::from("field_b"),
                FieldName::from("field_c"),
            ]),
            vec![
                DType::Bool(Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
                DType::Primitive(PType::I32, Nullability::Nullable),
            ],
        )
    }
}
