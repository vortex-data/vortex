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

use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaBuilder, SchemaRef};
use itertools::Itertools;
use vortex_datetime_dtype::arrow::{make_arrow_temporal_dtype, make_temporal_ext_dtype};
use vortex_datetime_dtype::is_temporal_ext_type;
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::arrow::{FromArrowType, TryFromArrowType};

impl TryFromArrowType<&DataType> for PType {
    fn try_from_arrow(value: &DataType) -> VortexResult<Self> {
        match value {
            DataType::Int8 => Ok(Self::I8),
            DataType::Int16 => Ok(Self::I16),
            DataType::Int32 => Ok(Self::I32),
            DataType::Int64 => Ok(Self::I64),
            DataType::UInt8 => Ok(Self::U8),
            DataType::UInt16 => Ok(Self::U16),
            DataType::UInt32 => Ok(Self::U32),
            DataType::UInt64 => Ok(Self::U64),
            DataType::Float16 => Ok(Self::F16),
            DataType::Float32 => Ok(Self::F32),
            DataType::Float64 => Ok(Self::F64),
            _ => Err(vortex_err!(
                "Arrow datatype {:?} cannot be converted to ptype",
                value
            )),
        }
    }
}

impl FromArrowType<SchemaRef> for DType {
    fn from_arrow(value: SchemaRef) -> Self {
        Self::Struct(
            StructDType::new(
                value
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str().into())
                    .collect_vec()
                    .into(),
                value
                    .fields()
                    .iter()
                    .map(|f| Self::from_arrow(f.as_ref()))
                    .collect_vec(),
            ),
            Nullability::NonNullable, // Must match From<RecordBatch> for Array
        )
    }
}

impl FromArrowType<&Field> for DType {
    fn from_arrow(field: &Field) -> Self {
        use vortex_dtype::DType::*;

        let nullability: Nullability = field.is_nullable().into();

        if let Ok(ptype) = PType::try_from_arrow(field.data_type()) {
            return Primitive(ptype, nullability);
        }

        match field.data_type() {
            DataType::Null => Null,
            DataType::Boolean => Bool(nullability),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Utf8(nullability),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Binary(nullability),
            DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(..) => Extension(Arc::new(
                make_temporal_ext_dtype(field.data_type()).with_nullability(nullability),
            )),
            DataType::List(e) | DataType::LargeList(e) => {
                List(Arc::new(Self::from_arrow(e.as_ref())), nullability)
            }
            DataType::Struct(f) => Struct(
                StructDType::new(
                    f.iter()
                        .map(|f| f.name().as_str().into())
                        .collect_vec()
                        .into(),
                    f.iter().map(|f| Self::from_arrow(f.as_ref())).collect_vec(),
                ),
                nullability,
            ),
            _ => unimplemented!("Arrow data type not yet supported: {:?}", field.data_type()),
        }
    }
}

/// Convert a Vortex [struct DType][DType] to an Arrow [Schema].
pub fn infer_schema(dtype: &DType) -> VortexResult<Schema> {
    let DType::Struct(struct_dtype, nullable) = dtype else {
        vortex_bail!("only DType::Struct can be converted to arrow schema");
    };

    if *nullable != Nullability::NonNullable {
        vortex_bail!("top-level struct in Schema must be NonNullable");
    }

    let mut builder = SchemaBuilder::with_capacity(struct_dtype.names().len());
    for (field_name, field_dtype) in struct_dtype
        .names()
        .iter()
        .zip(struct_dtype.dtypes().iter())
    {
        builder.push(FieldRef::from(Field::new(
            field_name.to_string(),
            infer_data_type(field_dtype)?,
            field_dtype.is_nullable(),
        )));
    }

    Ok(builder.finish())
}

/// Try to convert a Vortex [`DType`] into an Arrow [`DataType`]
pub fn infer_data_type(dtype: &DType) -> VortexResult<DataType> {
    Ok(match dtype {
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
        DType::Utf8(_) => DataType::Utf8View,
        DType::Binary(_) => DataType::BinaryView,
        DType::Struct(struct_dtype, _) => {
            let mut fields = Vec::with_capacity(struct_dtype.names().len());
            for (field_name, field_dt) in struct_dtype
                .names()
                .iter()
                .zip(struct_dtype.dtypes().iter())
            {
                fields.push(FieldRef::from(Field::new(
                    field_name.to_string(),
                    infer_data_type(field_dt)?,
                    field_dt.is_nullable(),
                )));
            }

            DataType::Struct(Fields::from(fields))
        }
        // There are four kinds of lists: List (32-bit offsets), Large List (64-bit), List View
        // (32-bit), Large List View (64-bit). We cannot both guarantee zero-copy and commit to an
        // Arrow dtype because we do not how large our offsets are.
        DType::List(..) => vortex_bail!("Unsupported dtype: {}", dtype),
        DType::Extension(ext_dtype) => {
            // Try and match against the known extension DTypes.
            if is_temporal_ext_type(ext_dtype.id()) {
                make_arrow_temporal_dtype(ext_dtype)
            } else {
                vortex_bail!("Unsupported extension type \"{}\"", ext_dtype.id())
            }
        }
    })
}

#[cfg(test)]
mod test {
    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};
    use vortex_dtype::{
        DType, ExtDType, ExtID, FieldName, FieldNames, Nullability, PType, StructDType,
    };

    use super::*;

    #[test]
    fn test_dtype_conversion_success() {
        assert_eq!(infer_data_type(&DType::Null).unwrap(), DataType::Null);

        assert_eq!(
            infer_data_type(&DType::Bool(Nullability::NonNullable)).unwrap(),
            DataType::Boolean
        );

        assert_eq!(
            infer_data_type(&DType::Primitive(PType::U64, Nullability::NonNullable)).unwrap(),
            DataType::UInt64
        );

        assert_eq!(
            infer_data_type(&DType::Utf8(Nullability::NonNullable)).unwrap(),
            DataType::Utf8View
        );

        assert_eq!(
            infer_data_type(&DType::Binary(Nullability::NonNullable)).unwrap(),
            DataType::BinaryView
        );

        assert_eq!(
            infer_data_type(&DType::Struct(
                StructDType::new(
                    FieldNames::from(vec![FieldName::from("field_a"), FieldName::from("field_b")]),
                    vec![DType::Bool(false.into()), DType::Utf8(true.into())],
                ),
                Nullability::NonNullable,
            ))
            .unwrap(),
            DataType::Struct(Fields::from(vec![
                FieldRef::from(Field::new("field_a", DataType::Boolean, false)),
                FieldRef::from(Field::new("field_b", DataType::Utf8View, true)),
            ]))
        );
    }

    #[test]
    #[should_panic]
    fn test_dtype_conversion_panics() {
        let _ = infer_data_type(&DType::Extension(Arc::new(ExtDType::new(
            ExtID::from("my-fake-ext-dtype"),
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            None,
        ))))
        .unwrap();
    }

    #[test]
    fn test_schema_conversion() {
        let struct_dtype = the_struct();
        let schema_nonnull = DType::Struct(struct_dtype, Nullability::NonNullable);

        assert_eq!(
            infer_schema(&schema_nonnull).unwrap(),
            Schema::new(Fields::from(vec![
                Field::new("field_a", DataType::Boolean, false),
                Field::new("field_b", DataType::Utf8View, false),
                Field::new("field_c", DataType::Int32, true),
            ]))
        );
    }

    #[test]
    #[should_panic]
    fn test_schema_conversion_panics() {
        let struct_dtype = the_struct();
        let schema_null = DType::Struct(struct_dtype, Nullability::Nullable);
        let _ = infer_schema(&schema_null).unwrap();
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
