//! Convert between Vortex [`DType`] and Apache Arrow [`DataType`].
//!
//! ## Arrow -> Vortex
//!
//! Each Arrow `DataType` has a defined mapping onto its nearest Vortex data type, via
//! implementation of the `FromArrowType` trait.
//!
//! All of the Arrow primitive types map onto the equivalent Vortex primitives:
//!
//! ```rust
//! use arrow_schema::DataType;
//! use vortex_dtype::{DType, Nullability, PType, arrow::FromArrowType};
//!
//! let arrow_i32 = DataType::Int32;
//! let vortex_i32 = DType::from_arrow((&arrow_i32, false.into()));
//! assert_eq!(vortex_i32, DType::Primitive(PType::I32, false.into()));
//! ```
//!
//! However, some types in Arrow do not map 1:1 onto Vortex. This is because
//! Arrow uses _physical_ type information, whereas Vortex is a pure logical
//! type system.For example, Arrow distinguishes between `String` and `LargeString`
//! layouts based on the size of the offset elements, whereas in Vortex they are
//! both just `Utf8`.
//!
//!
//! ```rust
//! use arrow_schema::DataType;
//! use vortex_dtype::{DType, Nullability, PType, arrow::FromArrowType};
//!
//! // This type has no exact representation in Vortex
//! let arrow_large_string = DataType::LargeUtf8;
//! let vortex_string = DType::from_arrow((&arrow_large_string, false.into()));
//! // The "Large" is lost in the conversion.
//! assert_eq!(vortex_string, DType::Utf8(false.into()));
//! ```
//!
//! There are many such cases where extra Arrow type information is lost. Users that
//! want to be able to round-trip back to Arrow later should save the original
//! `DataType`.
//!
//! ## Vortex -> Arrow
//!
//! `DType` has defined conversions into both [`Schema`] and [`DataType`], where the former
//! is only supported for struct types.
//!
//! [`DType::to_arrow_schema`] and its sibling [`DType::to_arrow`] follow a simple algorithm
//! for selecting the nearest Arrow type:
//!
//! * Vortex `Utf8` maps to Arrow `Utf8View` and Vortex `Binary` maps to `BinaryView`
//! * The non-`Large` variant of a type is always selected by default
//! * Extension types provide a mapping to
//!
//! ```rust
//! use std::sync::Arc;
//! use arrow_schema::DataType;
//! use vortex_dtype::{DType, Nullability, PType};
//!
//! // Utf8 will be selected over LargeUtf8
//! assert_eq!(
//!     DType::Utf8(false.into()).to_arrow().unwrap(),
//!     DataType::Utf8View,
//! );
//!
//! // List will be selected over LargeList
//! assert_eq!(
//!     DType::List(Arc::new(PType::I32.into()), false.into()).to_arrow().unwrap(),
//!     DataType::new_list(DataType::Int32, false),
//! );
//! ```
//!
//! ## Extension type support
//!
//! Arrow provides an extension type mechanism, effectively a type alias to another `DataType` with
//! some additional key-value string metadata that is carried on the field schema.
//!
//! Vortex supports canonicalizing its types into Arrow extension types, via the implementation
//! of the [`ArrowTypeConversion`] trait, and the [`register_extension_type!`][register_extension_type] macro.
//!
//! ```rust
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! use arcref::ArcRef;
//! use arrow_schema::Field;
//! use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
//! use vortex_dtype::{register_extension_type, DType, ExtDType, ExtID, PType, StructDType};
//! use vortex_dtype::arrow::{ArrowTypeConversion, ArrowTypeConversionRef};
//! use vortex_error::VortexResult;
//!
//! pub struct TemperatureConversion;
//!
//! impl ArrowTypeConversion for TemperatureConversion {
//!     fn arrow_metadata(&self, dtype: &ExtDType) -> VortexResult<Option<HashMap<String, String>>> {
//!         if dtype.id().as_ref() != "vortex.temperature" {
//!             return Ok(None);
//!         }
//!
//!         let mut metadata = HashMap::new();
//!         metadata.insert(EXTENSION_TYPE_NAME_KEY.to_string(), "vortex.temperature".to_string());
//!         metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_string(), r#"{"unit": "F"}"#.to_string());
//!         Ok(Some(metadata))
//!     }
//! }
//!
//! // Register the extension type so `DType` methods are aware of it at runtime.
//! register_extension_type! {
//!     ArrowTypeConversionRef::new(ArcRef::new_ref(&TemperatureConversion))
//! };
//!
//! // Attempt to use the extension type
//! let ext_type = ExtDType::new(
//!     ExtID::new("vortex.temperature".into()),
//!     Arc::new(DType::Primitive(PType::F32, false.into())),
//!     None,
//! );
//!
//! // Extension types only carry their metadata in a schema, so we
//! // use a struct type
//! let ext_type = DType::Extension(Arc::new(ext_type));
//! let schema = StructDType::from_fields(
//!     ["values".into()].into(),
//!     vec![ext_type.into()],
//! );
//!
//! let schema_type = DType::Struct(Arc::new(schema), false.into());
//!
//! let arrow_schema = schema_type.to_arrow_schema().unwrap();
//! let ext_field = arrow_schema.field(0);
//! assert_eq!(
//!     ext_field.extension_type_name(),
//!     Some("vortex.temperature"),
//! );
//!
//! assert_eq!(
//!     ext_field.extension_type_metadata(),
//!     Some(r#"{"unit": "F"}"#),
//! );
//! ```

use std::ops::Deref;
use std::sync::Arc;

use arcref::ArcRef;
use arrow_schema::{DECIMAL128_MAX_SCALE, DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};

use crate::datetime::arrow::{make_arrow_temporal_dtype, make_temporal_ext_dtype};
use crate::datetime::is_temporal_ext_type;
use crate::{DType, DecimalDType, ExtDType, FieldName, Nullability, PType, StructDType};

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
            DataType::Decimal128(precision, scale) | DataType::Decimal256(precision, scale) => {
                Decimal(DecimalDType::new(*precision, *scale), nullability)
            }
            DataType::Boolean => Bool(nullability),
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
        match field.extension_type_name() {
            None => Self::from_arrow((field.data_type(), field.is_nullable().into())),
            Some(ext_type) => {
                // Check the registry for any Vortex extension types that represent the Arrow
                // extension type named here.
                for converter in inventory::iter::<ArrowTypeConversionRef> {
                    if let Some(converted) = converter
                        .to_vortex(field)
                        .vortex_expect("Conversion from GeoArrow type to GeoVortex")
                    {
                        return converted;
                    }
                }

                // TODO(aduffy): should we just fallback to storage DType array and erase
                //  the type information instead? But if we do that, we lose ability to
                //  roundtrip back to Arrow.
                vortex_panic!(
                    "No supported conversion for Arrow extension type: {}",
                    ext_type
                )
            }
        }
    }
}

impl DType {
    /// Convert a Vortex [`DType`] into an Arrow [`Schema`].
    pub fn to_arrow_schema(&self) -> VortexResult<Schema> {
        let DataType::Struct(fields) = self.to_arrow()? else {
            vortex_bail!(
                "Cannot convert non-struct dtype to Arrow schema: {:?}",
                self
            )
        };
        Ok(Schema::new(fields))
    }

    /// Returns the Arrow [`Field`] that best represents the Vortex type.
    pub fn to_arrow(&self) -> VortexResult<DataType> {
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
                    let mut field = Field::new(
                        field_name.to_string(),
                        field_dt.to_arrow()?,
                        field_dt.is_nullable(),
                    );

                    // Optionally: attach any Arrow extension type metadata, if an ArrowMetadata
                    // kernel is defined for the extension type.
                    if let DType::Extension(ext_type) = field_dt {
                        for kernel in inventory::iter::<ArrowTypeConversionRef> {
                            if let Some(metadata) = kernel.0.arrow_metadata(ext_type.as_ref())? {
                                field.set_metadata(metadata);
                                break;
                            }
                        }
                    }

                    fields.push(field);
                }

                DataType::Struct(Fields::from(fields))
            }
            // There are four kinds of lists: List (32-bit offsets), Large List (64-bit), List View
            // (32-bit), Large List View (64-bit). We cannot both guarantee zero-copy and commit to an
            // Arrow dtype because we do not how large our offsets are.
            DType::List(elem_type, _) => DataType::List(FieldRef::new(Field::new_list_field(
                elem_type.to_arrow()?,
                elem_type.nullability().into(),
            ))),
            DType::Extension(ext_dtype) => {
                // Try and match against the known extension DTypes.
                if is_temporal_ext_type(ext_dtype.id()) {
                    make_arrow_temporal_dtype(ext_dtype)
                } else {
                    ext_dtype.storage_dtype().to_arrow()?
                }
            }
        })
    }
}

/// Attempt to convert a field to a DType. If the there is no registered converter that
/// can handle the field type, `None` is returned.
///
/// If a converter is resolved, it is used to convert the Field and the result is returned in
/// a `Some`.
pub fn geo_field_to_dtype(field: impl AsRef<Field>) -> VortexResult<Option<DType>> {
    for converter in inventory::iter::<ArrowTypeConversionRef> {
        if let Some(converted) = converter.to_vortex(field.as_ref())? {
            return Ok(Some(converted));
        }
    }

    Ok(None)
}

/// Conversions between Arrow [`Field`] type and Vortex logical `DType`.
///
/// This crate provides infra for registering and discovering extension types.
/// Once you implement this trait, you need to register it using [`register_extension_type`]
/// to have it get discovered.
///
/// See also: [`register_extension_type`]
pub trait ArrowTypeConversion: 'static + Send + Sync {
    /// Convert the given Arrow [`Field`] to a Vortex [`DType`].
    fn to_vortex(&self, _field: &Field) -> VortexResult<Option<DType>> {
        Ok(None)
    }

    /// Returns any Arrow metadata that should be added to the field when a field of this
    /// type is converted [into an Arrow data type][DType::to_arrow].
    ///
    /// This method should be implemented if you want to add support for a Vortex
    /// [extension type][ExtDType] that also has a corresponding canonical Arrow
    /// extension type, to preserve round-trip between the two.
    #[allow(clippy::disallowed_types)]
    fn arrow_metadata(
        &self,
        _dtype: &ExtDType,
    ) -> VortexResult<Option<std::collections::HashMap<String, String>>> {
        Ok(None)
    }
}

/// Conversion token
pub struct ArrowTypeConversionRef(ArcRef<dyn ArrowTypeConversion>);
inventory::collect!(ArrowTypeConversionRef);

impl Deref for ArrowTypeConversionRef {
    type Target = dyn ArrowTypeConversion;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl ArrowTypeConversionRef {
    /// Create a new `TypeConversionRef` from a pointer to an implementation.
    pub const fn new(conversion: ArcRef<dyn ArrowTypeConversion>) -> Self {
        Self(conversion)
    }
}

/// Register an Arrow extension type for lookup.
///
/// This is required to enable the type to be recognized by [`DType::from_arrow`]
/// as well as [`DType::to_arrow`].
#[macro_export]
macro_rules! register_extension_type {
    ($extension:expr) => {
        const _: $crate::arrow::ArrowTypeConversionRef = $extension;
        $crate::inventory::submit! { $extension }
    };
}

#[cfg(test)]
mod test {
    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};

    use super::*;
    use crate::{DType, ExtDType, ExtID, FieldName, FieldNames, Nullability, PType, StructDType};

    #[test]
    fn test_dtype_conversion_success() {
        assert_eq!(DType::Null.to_arrow().unwrap(), DataType::Null);

        assert_eq!(
            DType::Bool(Nullability::NonNullable).to_arrow().unwrap(),
            DataType::Boolean
        );

        assert_eq!(
            DType::Primitive(PType::U64, Nullability::NonNullable)
                .to_arrow()
                .unwrap(),
            DataType::UInt64
        );

        assert_eq!(
            DType::Utf8(Nullability::NonNullable).to_arrow().unwrap(),
            DataType::Utf8View
        );

        assert_eq!(
            DType::Binary(Nullability::NonNullable).to_arrow().unwrap(),
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
            .to_arrow()
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

        let arrow_list_non_nullable = list_non_nullable.to_arrow().unwrap();

        let list_nullable = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable,
        );
        let arrow_list_nullable = list_nullable.to_arrow().unwrap();

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
        .to_arrow()
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
