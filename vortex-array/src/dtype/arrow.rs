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
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::LEGACY_SESSION;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::session::DTypeSession;
use crate::dtype::session::DTypeSessionExt;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::Date;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;

const ARROW_EXT_NAME_VARIANT: &str = "arrow.parquet.variant";

/// Trait for converting Arrow types to Vortex types.
pub trait FromArrowType<T>: Sized {
    /// Convert the Arrow type to a Vortex type.
    fn from_arrow(value: T) -> Self;

    /// Convert the Arrow type to a Vortex type, consulting `session` for extension lookup.
    ///
    /// Unregistered or malformed extension metadata falls back to the storage dtype.
    fn from_arrow_with_session(value: T, _session: &VortexSession) -> Self {
        Self::from_arrow(value)
    }
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
        Self::from_arrow_with_session(value, &LEGACY_SESSION)
    }

    fn from_arrow_with_session(value: SchemaRef, session: &VortexSession) -> Self {
        Self::from_arrow_with_session(value.as_ref(), session)
    }
}

impl FromArrowType<&Schema> for DType {
    fn from_arrow(value: &Schema) -> Self {
        Self::from_arrow_with_session(value, &LEGACY_SESSION)
    }

    fn from_arrow_with_session(value: &Schema, session: &VortexSession) -> Self {
        Self::Struct(
            StructFields::from_arrow_with_session(value.fields(), session),
            Nullability::NonNullable, // Must match From<RecordBatch> for Array
        )
    }
}

impl FromArrowType<&Fields> for StructFields {
    fn from_arrow(value: &Fields) -> Self {
        Self::from_arrow_with_session(value, &LEGACY_SESSION)
    }

    fn from_arrow_with_session(value: &Fields, session: &VortexSession) -> Self {
        let dtypes = session.dtypes();
        StructFields::from_iter(value.into_iter().map(|f| {
            (
                FieldName::from(f.name().as_str()),
                dtype_from_field(f.as_ref(), &dtypes),
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
        Self::from_arrow_with_session(field, &LEGACY_SESSION)
    }

    fn from_arrow_with_session(field: &Field, session: &VortexSession) -> Self {
        dtype_from_field(field, &session.dtypes())
    }
}

/// Convert an Arrow Field to a [`DType`] with `dtypes` already borrowed from the session,
/// so the handle is acquired once per schema rather than once per field.
fn dtype_from_field(field: &Field, dtypes: &DTypeSession) -> DType {
    if field
        .extension_type_name()
        .is_some_and(|s| s == ARROW_EXT_NAME_VARIANT)
    {
        return DType::Variant(field.is_nullable().into());
    }

    let storage_dtype = storage_dtype_from_field(field, dtypes);
    match resolve_extension_dtype(field, dtypes, &storage_dtype) {
        Some(ext_ref) => DType::Extension(ext_ref),
        None => storage_dtype,
    }
}

/// Resolve the [`ExtDTypeRef`] for an Arrow Field whose `ARROW:extension:name` metadata names
/// a registered Vortex extension. Returns `None` for unregistered extensions, malformed
/// metadata, or fields with no extension name; callers fall back to the storage representation
/// and `tracing::warn!` reports the anomaly.
pub(crate) fn resolve_extension_dtype(
    field: &Field,
    dtypes: &DTypeSession,
    storage_dtype: &DType,
) -> Option<ExtDTypeRef> {
    let ext_name = field.extension_type_name()?;
    if ext_name == ARROW_EXT_NAME_VARIANT {
        return None;
    }

    let ext_id = ExtId::new(ext_name);
    let Some(plugin) = dtypes.registry().find(&ext_id) else {
        tracing::warn!(
            "Arrow field {:?} extension id {ext_name:?} not registered; using storage dtype",
            field.name(),
        );
        return None;
    };

    let metadata_bytes = match decode_extension_metadata(field) {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::warn!(
                "Arrow field {:?} extension id {ext_name:?} has malformed metadata ({e}); \
                 using storage dtype",
                field.name(),
            );
            return None;
        }
    };

    match plugin.deserialize(&metadata_bytes, storage_dtype.clone()) {
        Ok(ext_ref) => Some(ext_ref),
        Err(e) => {
            tracing::warn!(
                "Arrow field {:?} extension id {ext_name:?} failed to deserialize ({e}); \
                 using storage dtype",
                field.name(),
            );
            None
        }
    }
}

/// Extensions base64-encode arbitrary binary metadata to survive Arrow's String-typed
/// metadata channel.
fn decode_extension_metadata(field: &Field) -> VortexResult<Vec<u8>> {
    match field.extension_type_metadata() {
        None | Some("") => Ok(Vec::new()),
        Some(s) => BASE64_STANDARD
            .decode(s)
            .map_err(|e| vortex_err!("failed to base64-decode {EXTENSION_TYPE_METADATA_KEY}: {e}")),
    }
}

/// Build the storage [`DType`] for `field`, recursing through nested children so each level
/// runs the extension lookup against `dtypes`.
fn storage_dtype_from_field(field: &Field, dtypes: &DTypeSession) -> DType {
    let nullability: Nullability = field.is_nullable().into();
    match field.data_type() {
        DataType::Struct(f) => DType::Struct(
            StructFields::from_iter(f.into_iter().map(|child| {
                (
                    FieldName::from(child.name().as_str()),
                    dtype_from_field(child.as_ref(), dtypes),
                )
            })),
            nullability,
        ),
        DataType::List(e)
        | DataType::LargeList(e)
        | DataType::ListView(e)
        | DataType::LargeListView(e) => {
            DType::List(Arc::new(dtype_from_field(e.as_ref(), dtypes)), nullability)
        }
        DataType::FixedSizeList(e, size) => DType::FixedSizeList(
            Arc::new(dtype_from_field(e.as_ref(), dtypes)),
            *size as u32,
            nullability,
        ),
        other => DType::from_arrow((other, nullability)),
    }
}

impl DType {
    /// Convert a Vortex [`DType`] into an Arrow [`Schema`].
    pub fn to_arrow_schema(&self) -> VortexResult<Schema> {
        self.to_arrow_schema_with_session(&LEGACY_SESSION)
    }

    /// Convert a Vortex [`DType`] into an Arrow [`Schema`], threading `session` through nested
    /// fields so registered extensions are emitted with their `ARROW:extension:name` metadata.
    pub fn to_arrow_schema_with_session(&self, session: &VortexSession) -> VortexResult<Schema> {
        let DType::Struct(struct_dtype, nullable) = self else {
            vortex_bail!("only DType::Struct can be converted to arrow schema");
        };

        if *nullable != Nullability::NonNullable {
            vortex_bail!("top-level struct in Schema must be NonNullable");
        }

        let dtypes = session.dtypes();
        let mut builder = SchemaBuilder::with_capacity(struct_dtype.names().len());
        for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
            builder.push(field_from_dtype(
                field_name.as_ref(),
                &field_dtype,
                &dtypes,
            )?);
        }

        Ok(builder.finish())
    }

    /// Returns the Arrow [`DataType`] that best corresponds to this Vortex [`DType`].
    pub fn to_arrow_dtype(&self) -> VortexResult<DataType> {
        arrow_dtype_from_dtype(self, &LEGACY_SESSION.dtypes())
    }
}

fn arrow_dtype_from_dtype(dtype: &DType, dtypes: &DTypeSession) -> VortexResult<DataType> {
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
        DType::List(elem_dtype, _) => DataType::List(FieldRef::new(field_from_dtype(
            Field::LIST_FIELD_DEFAULT_NAME,
            elem_dtype,
            dtypes,
        )?)),
        DType::FixedSizeList(elem_dtype, size, _) => DataType::FixedSizeList(
            FieldRef::new(field_from_dtype(
                Field::LIST_FIELD_DEFAULT_NAME,
                elem_dtype,
                dtypes,
            )?),
            *size as i32,
        ),
        DType::Struct(struct_dtype, _) => {
            let mut fields = Vec::with_capacity(struct_dtype.names().len());
            for (field_name, field_dt) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
                fields.push(FieldRef::from(field_from_dtype(
                    field_name.as_ref(),
                    &field_dt,
                    dtypes,
                )?));
            }

            DataType::Struct(Fields::from(fields))
        }
        DType::Variant(_) => vortex_bail!(
            "DType::Variant requires Arrow Field metadata; use to_arrow_schema or a Field helper"
        ),
        DType::Extension(ext_dtype) => {
            if let Some(native) = native_arrow_dtype_for_extension(ext_dtype) {
                return Ok(native);
            }
            // Extension identity lives on the Field (see `field_from_dtype`), not on
            // DataType, so here we only encode the storage type.
            arrow_dtype_from_dtype(ext_dtype.storage_dtype(), dtypes)?
        }
    })
}

/// Build an Arrow [`Field`], attaching `ARROW:extension:name` and, when present,
/// `ARROW:extension:metadata` for extensions and Variant that have no native Arrow mapping.
fn field_from_dtype(name: &str, dtype: &DType, dtypes: &DTypeSession) -> VortexResult<Field> {
    if dtype.is_variant() {
        let storage = DataType::Struct(variant_storage_fields_minimal());
        return Ok(
            Field::new(name, storage, dtype.is_nullable()).with_metadata(
                [(
                    EXTENSION_TYPE_NAME_KEY.to_owned(),
                    ARROW_EXT_NAME_VARIANT.to_owned(),
                )]
                .into(),
            ),
        );
    }

    if let DType::Extension(ext) = dtype {
        // Native Arrow mapping carries the semantics in DataType; emitting extension metadata
        // on top would break consumers that only understand native Arrow types.
        if let Some(native) = native_arrow_dtype_for_extension(ext) {
            return Ok(Field::new(name, native, dtype.is_nullable()));
        }

        let storage_arrow = arrow_dtype_from_dtype(ext.storage_dtype(), dtypes)?;
        let ext_meta_bytes = ext.serialize_metadata()?;
        let meta_str = BASE64_STANDARD.encode(&ext_meta_bytes);

        let mut metadata = vec![(
            EXTENSION_TYPE_NAME_KEY.to_owned(),
            ext.id().as_str().to_owned(),
        )];
        if !meta_str.is_empty() {
            metadata.push((EXTENSION_TYPE_METADATA_KEY.to_owned(), meta_str));
        }
        return Ok(Field::new(name, storage_arrow, dtype.is_nullable())
            .with_metadata(metadata.into_iter().collect()));
    }

    Ok(Field::new(
        name,
        arrow_dtype_from_dtype(dtype, dtypes)?,
        dtype.is_nullable(),
    ))
}

/// Returns the native Arrow [`DataType`] for extensions Arrow models directly (e.g. temporal).
/// `None` means the extension should round-trip via storage + Field metadata.
fn native_arrow_dtype_for_extension(ext_dtype: &ExtDTypeRef) -> Option<DataType> {
    let temporal = ext_dtype.metadata_opt::<AnyTemporal>()?;
    Some(match temporal {
        TemporalMetadata::Timestamp(unit, tz) => {
            DataType::Timestamp(ArrowTimeUnit::try_from(*unit).ok()?, tz.clone())
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
    })
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
                .get(EXTENSION_TYPE_NAME_KEY)
                .map(|s| s.as_str()),
            Some(ARROW_EXT_NAME_VARIANT)
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

    mod extension_roundtrip {
        use vortex_session::VortexSession;

        use super::*;
        use crate::dtype::extension::ExtDType;
        use crate::dtype::session::DTypeSession;
        use crate::dtype::session::DTypeSessionExt;
        use crate::extension::tests::divisible_int::DivisibleInt;
        use crate::extension::tests::divisible_int::Divisor;

        fn session_with_divisible_int() -> VortexSession {
            let session = VortexSession::empty().with::<DTypeSession>();
            session.dtypes().register(DivisibleInt);
            session
        }

        fn divisible_ext(divisor: u64) -> DType {
            let ext = ExtDType::<DivisibleInt>::try_new(
                Divisor(divisor),
                DType::Primitive(PType::U64, Nullability::NonNullable),
            )
            .unwrap();
            DType::Extension(ext.erased())
        }

        #[test]
        fn forward_emits_name_and_base64_metadata() {
            let dtype = DType::struct_([("div", divisible_ext(7))], Nullability::NonNullable);

            let schema = dtype.to_arrow_schema().unwrap();
            let field = schema.field(0);

            assert_eq!(field.data_type(), &DataType::UInt64);
            assert_eq!(
                field
                    .metadata()
                    .get(EXTENSION_TYPE_NAME_KEY)
                    .map(String::as_str),
                Some("test.divisible_int"),
            );

            let meta_b64 = field.metadata().get(EXTENSION_TYPE_METADATA_KEY).unwrap();
            let decoded = BASE64_STANDARD.decode(meta_b64).unwrap();
            assert_eq!(decoded, 7u64.to_le_bytes());
        }

        #[test]
        fn reverse_with_session_recovers_extension() {
            let original = DType::struct_([("div", divisible_ext(42))], Nullability::NonNullable);

            let schema = original.to_arrow_schema().unwrap();
            let session = session_with_divisible_int();
            let recovered = DType::from_arrow_with_session(&schema, &session);

            assert_eq!(recovered, original);
        }

        #[test]
        fn reverse_without_registration_falls_back_to_storage() {
            let original = DType::struct_([("div", divisible_ext(13))], Nullability::NonNullable);

            let schema = original.to_arrow_schema().unwrap();
            // DivisibleInt is not in the default DTypeSession.
            let session = VortexSession::empty().with::<DTypeSession>();
            let recovered = DType::from_arrow_with_session(&schema, &session);

            let expected = DType::struct_(
                [(
                    "div",
                    DType::Primitive(PType::U64, Nullability::NonNullable),
                )],
                Nullability::NonNullable,
            );
            assert_eq!(recovered, expected);
        }

        #[test]
        fn nested_struct_roundtrip() {
            let inner = DType::struct_([("div", divisible_ext(3))], Nullability::Nullable);
            let original = DType::struct_([("inner", inner)], Nullability::NonNullable);

            let schema = original.to_arrow_schema().unwrap();
            let session = session_with_divisible_int();
            let recovered = DType::from_arrow_with_session(&schema, &session);

            assert_eq!(recovered, original);
        }

        #[test]
        fn list_element_roundtrip() {
            let list_dtype = DType::List(Arc::new(divisible_ext(5)), Nullability::Nullable);
            let original = DType::struct_([("xs", list_dtype)], Nullability::NonNullable);

            let schema = original.to_arrow_schema().unwrap();
            let session = session_with_divisible_int();
            let recovered = DType::from_arrow_with_session(&schema, &session);

            assert_eq!(recovered, original);
        }

        #[test]
        fn temporal_native_path_emits_no_extension_metadata() {
            let ts = Timestamp::new_with_tz(TimeUnit::Microseconds, None, Nullability::Nullable);
            let original = DType::struct_(
                [("t", DType::Extension(ts.erased()))],
                Nullability::NonNullable,
            );

            let schema = original.to_arrow_schema().unwrap();
            let field = schema.field(0);

            assert!(matches!(
                field.data_type(),
                DataType::Timestamp(ArrowTimeUnit::Microsecond, None)
            ));
            assert!(field.metadata().get(EXTENSION_TYPE_NAME_KEY).is_none());

            let recovered = DType::from_arrow(&schema);
            assert_eq!(recovered, original);
        }

        #[test]
        fn variant_still_roundtrips() {
            let original = DType::struct_(
                [("v", DType::Variant(Nullability::NonNullable))],
                Nullability::NonNullable,
            );
            let schema = original.to_arrow_schema().unwrap();
            let recovered = DType::from_arrow(&schema);
            assert_eq!(recovered, original);
        }

        #[test]
        fn malformed_metadata_falls_back_to_storage() {
            let field = Field::new("div", DataType::UInt64, false).with_metadata(
                [
                    (
                        EXTENSION_TYPE_NAME_KEY.to_owned(),
                        "test.divisible_int".to_owned(),
                    ),
                    (
                        EXTENSION_TYPE_METADATA_KEY.to_owned(),
                        "not_base64!!!".to_owned(),
                    ),
                ]
                .into(),
            );
            let schema = Schema::new(Fields::from(vec![field]));

            let session = session_with_divisible_int();
            let recovered = DType::from_arrow_with_session(&schema, &session);

            let expected = DType::struct_(
                [(
                    "div",
                    DType::Primitive(PType::U64, Nullability::NonNullable),
                )],
                Nullability::NonNullable,
            );
            assert_eq!(recovered, expected);
        }
    }
}
