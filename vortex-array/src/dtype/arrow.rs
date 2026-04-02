// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert between Vortex [`crate::dtype::DType`] and Apache Arrow [`arrow_schema::DataType`].
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

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::SchemaBuilder;
use arrow_schema::SchemaRef;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use arrow_schema::extension::ExtensionType as _;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::Date;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::uuid::Uuid;
use crate::extension::uuid::vtable::UUID_BYTE_LEN;

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
            DataType::Decimal32(precision, scale)
            | DataType::Decimal64(precision, scale)
            | DataType::Decimal128(precision, scale)
            | DataType::Decimal256(precision, scale) => Self::try_new(*precision, *scale),

            _ => Err(vortex_err!(
                "Arrow datatype {:?} cannot be converted to DecimalDType",
                value
            )),
        }
    }
}

impl From<&ArrowTimeUnit> for TimeUnit {
    fn from(value: &ArrowTimeUnit) -> Self {
        (*value).into()
    }
}

impl From<ArrowTimeUnit> for TimeUnit {
    fn from(value: ArrowTimeUnit) -> Self {
        match value {
            ArrowTimeUnit::Second => Self::Seconds,
            ArrowTimeUnit::Millisecond => Self::Milliseconds,
            ArrowTimeUnit::Microsecond => Self::Microseconds,
            ArrowTimeUnit::Nanosecond => Self::Nanoseconds,
        }
    }
}

impl TryFrom<TimeUnit> for ArrowTimeUnit {
    type Error = VortexError;

    fn try_from(value: TimeUnit) -> VortexResult<Self> {
        Ok(match value {
            TimeUnit::Seconds => Self::Second,
            TimeUnit::Milliseconds => Self::Millisecond,
            TimeUnit::Microseconds => Self::Microsecond,
            TimeUnit::Nanoseconds => Self::Nanosecond,
            _ => vortex_bail!("Cannot convert {value} to Arrow TimeUnit"),
        })
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
            StructFields::from_arrow(value.fields()),
            Nullability::NonNullable, // Must match From<RecordBatch> for Array
        )
    }
}

impl FromArrowType<&Fields> for StructFields {
    fn from_arrow(value: &Fields) -> Self {
        StructFields::from_iter(value.into_iter().map(|f| {
            (
                FieldName::from(f.name().as_str()),
                DType::from_arrow(f.as_ref()),
            )
        }))
    }
}

impl FromArrowType<(&DataType, Nullability)> for DType {
    fn from_arrow((data_type, nullability): (&DataType, Nullability)) -> Self {
        if data_type.is_integer() || data_type.is_floating() {
            return DType::Primitive(
                PType::try_from_arrow(data_type).vortex_expect("arrow float/integer to ptype"),
                nullability,
            );
        }

        match data_type {
            DataType::Null => DType::Null,
            DataType::Decimal32(precision, scale)
            | DataType::Decimal64(precision, scale)
            | DataType::Decimal128(precision, scale)
            | DataType::Decimal256(precision, scale) => {
                DType::Decimal(DecimalDType::new(*precision, *scale), nullability)
            }
            DataType::Boolean => DType::Bool(nullability),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DType::Utf8(nullability),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                DType::Binary(nullability)
            }
            DataType::Date32 => DType::Extension(Date::new(TimeUnit::Days, nullability).erased()),
            DataType::Date64 => {
                DType::Extension(Date::new(TimeUnit::Milliseconds, nullability).erased())
            }
            DataType::Time32(unit) => {
                DType::Extension(Time::new(unit.into(), nullability).erased())
            }
            DataType::Time64(unit) => {
                DType::Extension(Time::new(unit.into(), nullability).erased())
            }
            DataType::Timestamp(unit, tz) => DType::Extension(
                Timestamp::new_with_tz(unit.into(), tz.clone(), nullability).erased(),
            ),
            DataType::List(e)
            | DataType::LargeList(e)
            | DataType::ListView(e)
            | DataType::LargeListView(e) => {
                DType::List(Arc::new(Self::from_arrow(e.as_ref())), nullability)
            }
            DataType::FixedSizeList(e, size) => DType::FixedSizeList(
                Arc::new(Self::from_arrow(e.as_ref())),
                *size as u32,
                nullability,
            ),
            DataType::Struct(f) => DType::Struct(StructFields::from_arrow(f), nullability),
            DataType::Dictionary(_, value_type) => {
                Self::from_arrow((value_type.as_ref(), nullability))
            }
            DataType::RunEndEncoded(_, value_type) => {
                Self::from_arrow((value_type.data_type(), nullability))
            }
            _ => unimplemented!("Arrow data type not yet supported: {:?}", data_type),
        }
    }
}

impl FromArrowType<&Field> for DType {
    fn from_arrow(field: &Field) -> Self {
        let nullability = Nullability::from(field.is_nullable());

        if field
            .metadata()
            .get("ARROW:extension:name")
            .map(|s| s.as_str())
            == Some("arrow.parquet.variant")
        {
            return DType::Variant(nullability);
        }

        if field.extension_type_name() == Some(arrow_schema::extension::Uuid::NAME) {
            return DType::Extension(Uuid::default(nullability).erased());
        }

        Self::from_arrow((field.data_type(), nullability))
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
            let field = if field_dtype.is_variant() {
                let storage = DataType::Struct(variant_storage_fields_minimal());
                Field::new(field_name.as_ref(), storage, field_dtype.is_nullable()).with_metadata(
                    [(
                        "ARROW:extension:name".to_owned(),
                        "arrow.parquet.variant".to_owned(),
                    )]
                    .into(),
                )
            } else {
                let mut field = Field::new(
                    field_name.as_ref(),
                    field_dtype.to_arrow_dtype()?,
                    field_dtype.is_nullable(),
                );
                if let DType::Extension(ext) = field_dtype
                    && ext.is::<Uuid>()
                {
                    field = field.with_extension_type(arrow_schema::extension::Uuid);
                }
                field
            };
            builder.push(field);
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
                let precision = dt.precision();
                let scale = dt.scale();

                match precision {
                    // This code is commented out until DataFusion improves its support for smaller decimals.
                    // // DECIMAL32_MAX_PRECISION
                    // 0..=9 => DataType::Decimal32(precision, scale),
                    // // DECIMAL64_MAX_PRECISION
                    // 10..=18 => DataType::Decimal64(precision, scale),
                    // DECIMAL128_MAX_PRECISION
                    0..=38 => DataType::Decimal128(precision, scale),
                    // DECIMAL256_MAX_PRECISION
                    39.. => DataType::Decimal256(precision, scale),
                }
            }
            DType::Utf8(_) => DataType::Utf8View,
            DType::Binary(_) => DataType::BinaryView,
            // There are four kinds of lists: List (32-bit offsets), Large List (64-bit), List View
            // (32-bit), Large List View (64-bit). We cannot both guarantee zero-copy and commit to an
            // Arrow dtype because we do not how large our offsets are.
            DType::List(elem_dtype, _) => DataType::List(FieldRef::new(Field::new_list_field(
                elem_dtype.to_arrow_dtype()?,
                elem_dtype.nullability().into(),
            ))),
            DType::FixedSizeList(elem_dtype, size, _) => DataType::FixedSizeList(
                FieldRef::new(Field::new_list_field(
                    elem_dtype.to_arrow_dtype()?,
                    elem_dtype.nullability().into(),
                )),
                *size as i32,
            ),
            DType::Struct(struct_dtype, _) => {
                let mut fields = Vec::with_capacity(struct_dtype.names().len());
                for (field_name, field_dt) in struct_dtype.names().iter().zip(struct_dtype.fields())
                {
                    fields.push(FieldRef::from(Field::new(
                        field_name.as_ref(),
                        field_dt.to_arrow_dtype()?,
                        field_dt.is_nullable(),
                    )));
                }

                DataType::Struct(Fields::from(fields))
            }
            DType::Variant(_) => vortex_bail!(
                "DType::Variant requires Arrow Field metadata; use to_arrow_schema or a Field helper"
            ),
            DType::Extension(ext_dtype) => {
                // Try and match against the known extension DTypes.
                if let Some(temporal) = ext_dtype.metadata_opt::<AnyTemporal>() {
                    return Ok(match temporal {
                        TemporalMetadata::Timestamp(unit, tz) => {
                            DataType::Timestamp(ArrowTimeUnit::try_from(*unit)?, tz.clone())
                        }
                        TemporalMetadata::Date(unit) => match unit {
                            TimeUnit::Days => DataType::Date32,
                            TimeUnit::Milliseconds => DataType::Date64,
                            TimeUnit::Nanoseconds | TimeUnit::Microseconds | TimeUnit::Seconds => {
                                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", unit, ext_dtype.id())
                            }
                        },
                        TemporalMetadata::Time(unit) => match unit {
                            TimeUnit::Seconds => DataType::Time32(ArrowTimeUnit::Second),
                            TimeUnit::Milliseconds => DataType::Time32(ArrowTimeUnit::Millisecond),
                            TimeUnit::Microseconds => DataType::Time64(ArrowTimeUnit::Microsecond),
                            TimeUnit::Nanoseconds => DataType::Time64(ArrowTimeUnit::Nanosecond),
                            TimeUnit::Days => {
                                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", unit, ext_dtype.id())
                            }
                        },
                    });
                };

                if ext_dtype.is::<Uuid>() {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "UUID_BYTE_LEN always fits i32"
                    )]
                    return Ok(DataType::FixedSizeBinary(UUID_BYTE_LEN as i32));
                }

                vortex_bail!("Unsupported extension type \"{}\"", ext_dtype.id())
            }
        })
    }
}

fn variant_storage_fields_minimal() -> Fields {
    Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, true),
    ])
}

#[cfg(test)]
mod test {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::FieldRef;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;

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
            DType::struct_(
                [
                    ("field_a", DType::Bool(false.into())),
                    ("field_b", DType::Utf8(true.into()))
                ],
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
    fn test_variant_dtype_to_arrow_dtype_errors() {
        let err = DType::Variant(Nullability::NonNullable)
            .to_arrow_dtype()
            .unwrap_err()
            .to_string();
        assert!(err.contains("Variant"));
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
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
        );
        assert_eq!(
            arrow_list_non_nullable,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false))),
        );
    }

    #[fixture]
    fn the_struct() -> StructFields {
        StructFields::new(
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

    #[rstest]
    fn test_schema_conversion(the_struct: StructFields) {
        let schema_nonnull = DType::Struct(the_struct, Nullability::NonNullable);

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
    fn test_schema_variant_field_metadata() {
        let dtype = DType::struct_(
            [("v", DType::Variant(Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        let schema = dtype.to_arrow_schema().unwrap();
        let field = schema.field(0);
        assert_eq!(
            field
                .metadata()
                .get("ARROW:extension:name")
                .map(|s| s.as_str()),
            Some("arrow.parquet.variant")
        );
        assert!(matches!(field.data_type(), DataType::Struct(_)));
        assert!(!field.is_nullable());
    }

    #[rstest]
    #[should_panic]
    fn test_schema_conversion_panics(the_struct: StructFields) {
        let schema_null = DType::Struct(the_struct, Nullability::Nullable);
        schema_null.to_arrow_schema().unwrap();
    }

    #[test]
    fn test_unicode_field_names_roundtrip() {
        // Regression test for https://github.com/vortex-data/vortex/issues/5979.

        // Unicode characters in field names should survive an Arrow roundtrip without
        // double-escaping.
        let unicode_field_name = "\u{5}=A";
        let original_dtype = DType::struct_(
            [(
                unicode_field_name,
                DType::Primitive(PType::I8, Nullability::Nullable),
            )],
            Nullability::NonNullable,
        );

        let arrow_dtype = original_dtype.to_arrow_dtype().unwrap();
        let roundtripped_dtype = DType::from_arrow((&arrow_dtype, Nullability::NonNullable));

        assert_eq!(original_dtype, roundtripped_dtype);
    }

    #[test]
    fn test_unicode_field_names_nested_roundtrip() {
        // Regression test for https://github.com/vortex-data/vortex/issues/5979.

        // Nested structs with unicode field names should also survive an Arrow roundtrip.
        let inner_struct = DType::struct_(
            [(
                "\u{6}=inner",
                DType::Primitive(PType::I32, Nullability::Nullable),
            )],
            Nullability::Nullable,
        );
        let original_dtype =
            DType::struct_([("\u{7}=outer", inner_struct)], Nullability::NonNullable);

        let arrow_dtype = original_dtype.to_arrow_dtype().unwrap();
        let roundtripped_dtype = DType::from_arrow((&arrow_dtype, Nullability::NonNullable));

        assert_eq!(original_dtype, roundtripped_dtype);
    }

    #[test]
    fn test_uuid_schema_roundtrip() {
        let original = DType::struct_(
            [(
                "id",
                DType::Extension(Uuid::default(Nullability::Nullable).erased()),
            )],
            Nullability::NonNullable,
        );
        let schema = original.to_arrow_schema().unwrap();

        let field = schema.field(0);
        assert_eq!(field.data_type(), &DataType::FixedSizeBinary(16));
        assert_eq!(
            field.extension_type_name(),
            Some(arrow_schema::extension::Uuid::NAME)
        );

        assert_eq!(DType::from_arrow(&schema), original);
    }
}
