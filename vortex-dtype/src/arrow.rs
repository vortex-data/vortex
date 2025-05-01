//! Convert between Vortex [`crate::DType`] and Apache Arrow [`arrow_schema::DataType`].
//!
//! Apache Arrow's type system includes physical information, which could lead to ambiguities as
//! Vortex treats encodings as separate from logical types.
//!
//! [`DType::to_arrow_schema`] and its sibling [`DType::to_arrow_dtype`] use a simple algorithm,
//! where every logical type is encoded in its simplest corresponding Arrow type. This reflects the
//! reality that most compute engines don't make use of the entire type range arrow-rs supports.
//!
//! For this reason, it's recommended to do as much computation as possible within Vortex, and then
//! materialize an Arrow ArrayRef at the very end of the processing chain.

use std::sync::Arc;

use arrow_schema::{
    DECIMAL128_MAX_SCALE, DataType, Field, FieldRef, Fields, Schema, SchemaBuilder, SchemaRef,
};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::datetime::arrow::{make_arrow_temporal_dtype, make_temporal_ext_dtype};
use crate::datetime::is_temporal_ext_type;
use crate::{DType, DecimalDType, FieldName, Nullability, PType, StructDType};

/// Trait for converting Arrow types to Vortex types.
pub trait FromArrowType<T>: Sized {
    /// Convert the Arrow type to a Vortex type.
    fn from_arrow(value: T) -> Self;
}

/// Trait for converting Vortex types to Arrow types.
pub trait TryFromArrowType<T>: Sized {
    /// Convert the Arrow type to a Vortex type.
    fn try_from_arrow(value: T) -> VortexResult<Self>;
}

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

impl TryFromArrowType<&DataType> for DecimalDType {
    fn try_from_arrow(value: &DataType) -> VortexResult<Self> {
        match value {
            DataType::Decimal128(precision, scale) => Ok(Self::new(*precision, *scale)),
            DataType::Decimal256(precision, scale) => Ok(Self::new(*precision, *scale)),
            _ => Err(vortex_err!(
                "Arrow datatype {:?} cannot be converted to DecimalDType",
                value
            )),
        }
    }
}

impl FromArrowType<SchemaRef> for DType {
    fn from_arrow(value: SchemaRef) -> Self {
        Self::from_arrow(value.as_ref())
    }
}

impl FromArrowType<&Schema> for DType {
    fn from_arrow(value: &Schema) -> Self {
        Self::Struct(
            Arc::new(StructDType::from_arrow(value.fields())),
            Nullability::NonNullable, // Must match From<RecordBatch> for Array
        )
    }
}

impl FromArrowType<&Fields> for StructDType {
    fn from_arrow(value: &Fields) -> Self {
        StructDType::from_iter(value.into_iter().map(|f| {
            (
                FieldName::from(f.name().as_str()),
                DType::from_arrow(f.as_ref()),
            )
        }))
    }
}

impl FromArrowType<(&DataType, Nullability)> for DType {
    fn from_arrow((data_type, nullability): (&DataType, Nullability)) -> Self {
        use crate::DType::*;

        if data_type.is_integer() || data_type.is_floating() {
            return Primitive(
                PType::try_from_arrow(data_type).vortex_expect("arrow float/integer to ptype"),
                nullability,
            );
        }

        match data_type {
            DataType::Null => Null,
            DataType::Boolean => Bool(nullability),
            DataType::Decimal128(p, s) => Decimal(DecimalDType::new(*p, *s), nullability),
            DataType::Decimal256(p, s) => Decimal(DecimalDType::new(*p, *s), nullability),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Utf8(nullability),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Binary(nullability),
            DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(..) => Extension(Arc::new(
                make_temporal_ext_dtype(data_type).with_nullability(nullability),
            )),
            DataType::List(e) | DataType::LargeList(e) => {
                List(Arc::new(Self::from_arrow(e.as_ref())), nullability)
            }
            DataType::Struct(f) => Struct(Arc::new(StructDType::from_arrow(f)), nullability),
            _ => unimplemented!("Arrow data type not yet supported: {:?}", data_type),
        }
    }
}

impl FromArrowType<&Field> for DType {
    fn from_arrow(field: &Field) -> Self {
        Self::from_arrow((field.data_type(), field.is_nullable().into()))
    }
}

impl DType {
    /// Convert a Vortex [`DType`] into an Arrow [`Schema`].
    pub fn to_arrow_schema(&self) -> VortexResult<Schema> {
        let DType::Struct(struct_dtype, nullable) = self else {
            vortex_bail!("only DType::Struct can be converted to arrow schema");
        };

        if *nullable != Nullability::NonNullable {
            vortex_bail!("top-level struct in Schema must be NonNullable");
        }

        let mut builder = SchemaBuilder::with_capacity(struct_dtype.names().len());
        for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
            builder.push(FieldRef::from(Field::new(
                field_name.to_string(),
                field_dtype.to_arrow_dtype()?,
                field_dtype.is_nullable(),
            )));
        }

        Ok(builder.finish())
    }

    /// Returns the Arrow [`DataType`] that best corresponds to this Vortex [`DType`].
    pub fn to_arrow_dtype(&self) -> VortexResult<DataType> {
        Ok(match self {
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
            DType::Decimal(dt, _) => {
                if dt.scale() > DECIMAL128_MAX_SCALE {
                    DataType::Decimal256(dt.precision(), dt.scale())
                } else {
                    DataType::Decimal128(dt.precision(), dt.scale())
                }
            }
            DType::Utf8(_) => DataType::Utf8View,
            DType::Binary(_) => DataType::BinaryView,
            DType::Struct(struct_dtype, _) => {
                let mut fields = Vec::with_capacity(struct_dtype.names().len());
                for (field_name, field_dt) in struct_dtype.names().iter().zip(struct_dtype.fields())
                {
                    fields.push(FieldRef::from(Field::new(
                        field_name.to_string(),
                        field_dt.to_arrow_dtype()?,
                        field_dt.is_nullable(),
                    )));
                }

                DataType::Struct(Fields::from(fields))
            }
            // There are four kinds of lists: List (32-bit offsets), Large List (64-bit), List View
            // (32-bit), Large List View (64-bit). We cannot both guarantee zero-copy and commit to an
            // Arrow dtype because we do not how large our offsets are.
            DType::List(l, _) => DataType::List(FieldRef::new(Field::new_list_field(
                l.to_arrow_dtype()?,
                l.nullability().into(),
            ))),
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
}

#[cfg(test)]
mod test {
    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};

    use super::*;
    use crate::{DType, ExtDType, ExtID, FieldName, FieldNames, Nullability, PType, StructDType};

    #[test]
    fn test_dtype_conversion_success() {
        assert_eq!(DType::Null.to_arrow_dtype().unwrap(), DataType::Null);

        assert_eq!(
            DType::Bool(Nullability::NonNullable)
                .to_arrow_dtype()
                .unwrap(),
            DataType::Boolean
        );

        assert_eq!(
            DType::Primitive(PType::U64, Nullability::NonNullable)
                .to_arrow_dtype()
                .unwrap(),
            DataType::UInt64
        );

        assert_eq!(
            DType::Utf8(Nullability::NonNullable)
                .to_arrow_dtype()
                .unwrap(),
            DataType::Utf8View
        );

        assert_eq!(
            DType::Binary(Nullability::NonNullable)
                .to_arrow_dtype()
                .unwrap(),
            DataType::BinaryView
        );

        assert_eq!(
            DType::Struct(
                Arc::new(StructDType::from_iter([
                    ("field_a", DType::Bool(false.into())),
                    ("field_b", DType::Utf8(true.into()))
                ])),
                Nullability::NonNullable,
            )
            .to_arrow_dtype()
            .unwrap(),
            DataType::Struct(Fields::from(vec![
                FieldRef::from(Field::new("field_a", DataType::Boolean, false)),
                FieldRef::from(Field::new("field_b", DataType::Utf8View, true)),
            ]))
        );
    }

    #[test]
    fn infer_nullable_list_element() {
        let list_non_nullable = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::Nullable,
        );

        let arrow_list_non_nullable = list_non_nullable.to_arrow_dtype().unwrap();

        let list_nullable = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable,
        );
        let arrow_list_nullable = list_nullable.to_arrow_dtype().unwrap();

        assert_ne!(arrow_list_non_nullable, arrow_list_nullable);
        assert_eq!(
            arrow_list_nullable,
            DataType::new_list(DataType::Int64, true)
        );
        assert_eq!(
            arrow_list_non_nullable,
            DataType::new_list(DataType::Int64, false)
        );
    }

    #[test]
    #[should_panic]
    fn test_dtype_conversion_panics() {
        let _ = DType::Extension(Arc::new(ExtDType::new(
            ExtID::from("my-fake-ext-dtype"),
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            None,
        )))
        .to_arrow_dtype()
        .unwrap();
    }

    #[test]
    fn test_schema_conversion() {
        let struct_dtype = the_struct();
        let schema_nonnull = DType::Struct(struct_dtype, Nullability::NonNullable);

        assert_eq!(
            schema_nonnull.to_arrow_schema().unwrap(),
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
        let _ = schema_null.to_arrow_schema().unwrap();
    }

    fn the_struct() -> Arc<StructDType> {
        Arc::new(StructDType::new(
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
        ))
    }
}
