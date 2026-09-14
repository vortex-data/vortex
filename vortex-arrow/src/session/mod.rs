// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plugin layer for moving Arrow types Vortex cannot convert on its own in and out of Vortex.
//!
//! Vortex's canonical Arrow conversion (see [`crate::dtype`] and the executor in
//! [`crate::executor`]) handles every non-extension Arrow type and the builtin temporal
//! extensions, by executing an array to a canonical encoding and mapping that to Arrow. The
//! plugins registered here cover what that cannot do: **Arrow extension types**, and **encodings
//! that reach Arrow better on their own** than by being canonicalized first.
//!
//! * An [`ArrowExportVTable`] converts a Vortex array to Arrow. Its [`ArrowExportKey`] names the
//!   **source** it exports from — a Vortex extension dtype, or an encoding — and optionally the
//!   **target** it exports to: an Arrow extension name, or an exact [`DataType`]. An export is
//!   dispatched by matching the array's dtype and encoding against the former, and the requested
//!   [`Field`] against the latter, most specific first.
//! * An [`ArrowImportVTable`] is dispatched by the **source Arrow extension name** carried
//!   on the incoming [`Field`]. The plugin is responsible for both preserving extension
//!   identity and re-encoding storage if needed (e.g. Arrow `FixedSizeBinary[16]` for UUID
//!   becomes Vortex `FixedSizeList<u8; 16>`).
//!
//! Multiple plugins may register against the same key. They are tried in registration order;
//! each may return [`ArrowExport::Unsupported`] / [`ArrowImport::Unsupported`] to defer to
//! the next.

use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::RunArray;
use arrow_array::make_array;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::RunEndIndexType;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use tracing::trace;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtId;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrowExportOptions;
use crate::IntoVortexArray;
use crate::convert::from_arrow_dyn;
use crate::convert::map_from_arrow_parts;
use crate::convert::nulls;
use crate::convert::remove_nulls;
use crate::convert::trim_offsets;
use crate::dtype::from_arrow_data_type;
use crate::dtype::no_arrow_type;
use crate::dtype::to_data_type_naive;
use crate::executor::execute_arrow_naive;
use crate::run_end_import::run_end_from_arrow;

/// Outcome of a successful call to [`ArrowExportVTable::execute_arrow`].
///
/// Plugins that don't handle the supplied array return [`Unsupported`][Self::Unsupported]
/// with ownership of the input so the session can probe the next plugin or fall back to the
/// canonical path. Errors are propagated through [`VortexResult`].
pub enum ArrowExport {
    /// The plugin does not handle this input; the session may try another plugin.
    Unsupported(ArrayRef),
    /// A successful export.
    Exported(ArrowArrayRef),
}

/// Outcome of a successful call to [`ArrowImportVTable::from_arrow_array`].
///
/// Plugins that don't handle the supplied array return [`Unsupported`][Self::Unsupported]
/// with ownership of the input so the session can probe the next plugin or fall back to the
/// canonical path. Errors are propagated through [`VortexResult`].
pub enum ArrowImport {
    /// The plugin does not handle this input; the session may try another plugin.
    Unsupported(ArrowArrayRef),
    /// A successful import.
    Imported(ArrayRef),
}

/// The Arrow type description of an array being imported by [`ArrowSession::from_arrow_array`].
///
/// Callers holding an Arrow [`Field`] (or [`FieldRef`]) should pass it: its
/// `ARROW:extension:name` metadata is what dispatches the array to a registered
/// [`ArrowImportVTable`].
///
/// An Arrow array can carry a validity (null) buffer regardless of whether its schema declares
/// the field nullable, so when no [`Field`] is in hand the caller passes the desired nullability
/// instead, as a [`bool`] or a [`Nullability`]. An anonymous field is then synthesized from the
/// array's own data type, which means no extension plugin is dispatched for the array itself;
/// fields nested inside container data types still carry their metadata and are routed through
/// their importers.
pub trait IntoArrowField<'a> {
    /// Resolve to the Arrow [`Field`] describing an array of `data_type`.
    fn into_arrow_field(self, data_type: &DataType) -> Cow<'a, Field>;
}

impl<'a> IntoArrowField<'a> for &'a Field {
    fn into_arrow_field(self, _data_type: &DataType) -> Cow<'a, Field> {
        Cow::Borrowed(self)
    }
}

impl<'a> IntoArrowField<'a> for &'a FieldRef {
    fn into_arrow_field(self, _data_type: &DataType) -> Cow<'a, Field> {
        Cow::Borrowed(self.as_ref())
    }
}

impl<'a> IntoArrowField<'a> for bool {
    fn into_arrow_field(self, data_type: &DataType) -> Cow<'a, Field> {
        Cow::Owned(Field::new("", data_type.clone(), self))
    }
}

impl<'a> IntoArrowField<'a> for Nullability {
    fn into_arrow_field(self, data_type: &DataType) -> Cow<'a, Field> {
        self.is_nullable().into_arrow_field(data_type)
    }
}

/// What an [`ArrowExportVTable`] exports from: matched against the source array.
///
/// A plugin may also name no source — see [`ArrowExportKey::to_extension`] — and be dispatched
/// on its target alone. Such a plugin checks the array itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExportSource {
    /// Arrays of this Vortex extension dtype, whatever their encoding.
    Extension(ExtId),
    /// Arrays of this encoding, whatever their dtype.
    Encoding(ArrayId),
}

impl ExportSource {
    /// The sources an array is dispatched under, most specific first: its extension dtype, if it
    /// has one, then its encoding, then `None` for plugins that named no source.
    fn of(array: &ArrayRef) -> impl Iterator<Item = Option<Self>> + use<> {
        let extension = array
            .dtype()
            .as_extension_opt()
            .map(|ext| Self::Extension(ext.id()));
        [extension, Some(Self::Encoding(array.encoding_id())), None].into_iter()
    }
}

/// What an [`ArrowExportVTable`] exports to: matched against the target [`Field`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// A field carrying this `ARROW:extension:name`.
    Extension(Id),
    /// Exactly this Arrow type, parameters included: a plugin producing `Dictionary(UInt8, Utf8)`
    /// is not dispatched for a `Dictionary(UInt16, Utf8)` export.
    DataType(DataType),
}

/// How an [`ArrowExportVTable`] is dispatched: the source it exports from and the target it
/// exports to, at least one of which it names.
///
/// A plugin naming no target claims every export from its source, including one that requested
/// no particular Arrow type — which such a plugin then chooses for itself. Plugins naming a
/// target are tried first. A plugin naming no source is tried for every array exported to its
/// target, after those naming the array's dtype or encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrowExportKey {
    source: Option<ExportSource>,
    target: Option<ExportTarget>,
}

impl ArrowExportKey {
    /// Dispatch by `source`, for exports to `target` — or to anything, when `None`.
    pub fn new(source: ExportSource, target: Option<ExportTarget>) -> Self {
        Self {
            source: Some(source),
            target,
        }
    }

    /// Export to the Arrow extension `arrow_ext_id`, whatever the source array. For a logical
    /// type with no [`ExportSource`] of its own, such as [`DType::Variant`].
    pub fn to_extension(arrow_ext_id: Id) -> Self {
        Self {
            source: None,
            target: Some(ExportTarget::Extension(arrow_ext_id)),
        }
    }

    /// Export the Vortex extension `vortex_ext_id` to the Arrow extension `arrow_ext_id`.
    pub fn extension(vortex_ext_id: ExtId, arrow_ext_id: Id) -> Self {
        Self::new(
            ExportSource::Extension(vortex_ext_id),
            Some(ExportTarget::Extension(arrow_ext_id)),
        )
    }

    /// Export the Vortex encoding `encoding_id` to every Arrow type.
    pub fn encoding(encoding_id: ArrayId) -> Self {
        Self::new(ExportSource::Encoding(encoding_id), None)
    }

    /// Export the Vortex encoding `encoding_id` to `data_type` only.
    pub fn encoding_to(encoding_id: ArrayId, data_type: DataType) -> Self {
        Self::new(
            ExportSource::Encoding(encoding_id),
            Some(ExportTarget::DataType(data_type)),
        )
    }
}

/// Plugin layer for exporting a Vortex array to Arrow.
///
/// A plugin covers a conversion the canonical Arrow conversion cannot do: producing an Arrow
/// extension type, or exporting an encoding directly instead of letting it be executed to a
/// canonical encoding first — which would throw away the layout the encoding could have exported.
/// [`export_key`][Self::export_key] picks which of the two it is; see [`ArrowExportKey`].
///
/// Encoding-keyed plugins are dispatched on the encoding of the array as it stands, without
/// executing it first. A plugin that also wants to claim arrays its encoding is buried under —
/// behind a lazy `filter` or `slice`, say — should execute towards its own encoding itself, with
/// [`ArrayRef::execute_until`](vortex_array::ArrayRef::execute_until).
///
/// This is purely an implementation trait, its methods should not be called directly. Instead,
/// use the methods on [`ArrowSession`].
pub trait ArrowExportVTable: 'static + Send + Sync + Debug {
    /// How this plugin is dispatched, and the key it registers under.
    fn export_key(&self) -> ArrowExportKey;

    /// Build the Arrow [`Field`] this plugin produces for the given Vortex extension
    /// `dtype`. Used during schema inference.
    ///
    /// Only consulted for plugins whose [`ExportSource`] is a Vortex extension; the default
    /// declines, which is what an encoding-keyed plugin wants.
    fn to_arrow_field(
        &self,
        _name: &str,
        _dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        Ok(None)
    }

    /// Convert a Vortex array into an Arrow array shaped to `target`.
    ///
    /// `target` is `None` only for a plugin that named no [`ExportTarget`], and only when the
    /// export requested no particular Arrow type: the plugin then picks the type its source
    /// reaches most cheaply. Every other plugin is dispatched by something the target carries, so
    /// it always receives one, and must produce exactly its [`DataType`]. Either way the result
    /// must have the same length as `array`.
    ///
    /// Returns ownership of `array` via [`ArrowExport::Unsupported`] when the plugin cannot
    /// handle the input, which defers to the next plugin registered under the same key, and
    /// ultimately to the canonical conversion. Plugins that export child or storage arrays must
    /// pass `options` on to those exports.
    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport>;
}

/// Plugin layer for importing an Arrow extension-typed array into a Vortex array.
///
/// Plugins are dispatched by `arrow_ext_id`.
///
/// This is purely an implementation trait, its methods should not be called directly. Instead,
/// use the methods on [`ArrowSession`].
pub trait ArrowImportVTable: 'static + Send + Sync + Debug {
    /// The Arrow extension name this plugin handles.
    fn arrow_ext_id(&self) -> Id;

    /// Build the Vortex [`DType`] that corresponds to `field` (which carries this plugin's
    /// Arrow extension metadata).
    ///
    /// `session` is provided so plugins can resolve nested or storage fields through the
    /// plugin-aware conversion (e.g. [`ArrowSession::from_arrow_datatype`]) instead of the
    /// naive Arrow → Vortex mapping.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>>;

    /// Convert an Arrow array into a Vortex array of `dtype`.
    ///
    /// Returns ownership of `array` via [`ArrowImport::Unsupported`] when the plugin cannot
    /// handle the input.
    ///
    /// `session` is provided so plugins can convert storage or nested arrays through the
    /// session (e.g. [`ArrowSession::from_arrow_array`], passing a nullability rather than a
    /// [`Field`] so the plugin is not dispatched again) instead of the deprecated
    /// `FromArrowArray` trait.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport>;
}

pub type ArrowExportVTableRef = Arc<dyn ArrowExportVTable>;
pub type ArrowImportVTableRef = Arc<dyn ArrowImportVTable>;

/// Registry of Arrow importers, keyed by source Arrow extension [`Id`].
type ArrowImporterRegistry = ArcSwapMap<Id, Arc<[ArrowImportVTableRef]>>;
/// Registry of Arrow exporters, keyed by the source they export from, `None` for plugins that
/// named no source.
type ArrowExporterRegistry = ArcSwapMap<Option<ExportSource>, Arc<Exporters>>;

/// The exporters registered for one [`ExportSource`], split by the target they export to.
#[derive(Clone, Debug, Default)]
struct Exporters {
    /// Plugins producing a field carrying this `ARROW:extension:name`.
    by_extension: HashMap<Id, Vec<ArrowExportVTableRef>>,
    /// Plugins producing exactly this Arrow type.
    by_data_type: HashMap<DataType, Vec<ArrowExportVTableRef>>,
    /// Plugins claiming every export from the source.
    any: Vec<ArrowExportVTableRef>,
}

impl Exporters {
    fn push(&mut self, target: Option<ExportTarget>, exporter: ArrowExportVTableRef) {
        let plugins = match target {
            Some(ExportTarget::Extension(id)) => self.by_extension.entry(id).or_default(),
            Some(ExportTarget::DataType(data_type)) => {
                self.by_data_type.entry(data_type).or_default()
            }
            None => &mut self.any,
        };
        plugins.push(exporter);
    }

    /// Every plugin registered for the source, those naming a target first.
    fn all(&self) -> impl Iterator<Item = &ArrowExportVTableRef> {
        self.by_extension
            .values()
            .chain(self.by_data_type.values())
            .flatten()
            .chain(&self.any)
    }
}

/// Session-scoped registry of Arrow plugins.
///
/// Exporters are indexed by their [`ExportSource`] and, within a source, by their
/// [`ExportTarget`]. Dispatch walks the array's sources most specific first (its extension dtype,
/// its encoding, then plugins naming no source) and, for each, the targets most specific first
/// (the Arrow extension name asked for, the exact Arrow type, then plugins claiming any type).
/// Importers are keyed by Arrow extension name.
///
/// Extension plugins are registered by the crate that owns the extension type (for example
/// `vortex-uuid`); temporal extensions are handled by the canonical Arrow ↔ Vortex path and do not
/// need plugins.
#[derive(Clone, Debug, Default)]
pub struct ArrowSession {
    exporters: ArrowExporterRegistry,
    importers: ArrowImporterRegistry,
}

impl ArrowSession {
    /// Register an [`ArrowExportVTable`] under its [`ArrowExportKey`]. Plugins sharing a key are
    /// tried in registration order.
    pub fn register_exporter(&self, exporter: ArrowExportVTableRef) {
        let ArrowExportKey { source, target } = exporter.export_key();
        self.exporters.update(source, |existing| {
            let mut exporters = existing.map(|e| Exporters::clone(e)).unwrap_or_default();
            exporters.push(target.clone(), ArrowExportVTableRef::clone(&exporter));
            Arc::new(exporters)
        });
    }

    /// Register an [`ArrowImportVTable`] under its source Arrow extension name.
    pub fn register_importer(&self, importer: ArrowImportVTableRef) {
        self.importers.push(importer.arrow_ext_id(), importer);
    }

    fn exporters(&self, source: Option<ExportSource>) -> Arc<Exporters> {
        self.exporters.get(&source).unwrap_or_default()
    }

    fn importers(&self, id: &Id) -> Arc<[ArrowImportVTableRef]> {
        self.importers.get(id).unwrap_or_else(|| Arc::from([]))
    }

    /// Build the Arrow [`Field`] for a Vortex [`DType`], or `None` if the dtype has no Arrow
    /// representation.
    ///
    /// For [`DType::Extension`]s, plugins registered against the extension's `Id`
    /// are tried in registration order; the first plugin to return `Some(field)` wins.
    pub fn to_arrow_field(&self, name: &str, dtype: &DType) -> VortexResult<Option<Field>> {
        // Handle the structural encodings, which may have recursive types
        let field = match dtype {
            DType::List(elem_dtype, nullability) => {
                let Some(elem_field) =
                    self.to_arrow_field(Field::LIST_FIELD_DEFAULT_NAME, elem_dtype)?
                else {
                    return Ok(None);
                };

                Field::new_list(name, elem_field, nullability.is_nullable())
            }
            DType::FixedSizeList(elem_dtype, elem_size, nullability) => {
                let Some(elem_field) =
                    self.to_arrow_field(Field::LIST_FIELD_DEFAULT_NAME, elem_dtype)?
                else {
                    return Ok(None);
                };

                Field::new_fixed_size_list(
                    name,
                    elem_field,
                    (*elem_size).try_into()?,
                    nullability.is_nullable(),
                )
            }
            DType::Map(map_dtype, nullability) => {
                let (Some(key), Some(value)) = (
                    self.to_arrow_field("key", &map_dtype.key_dtype())?,
                    self.to_arrow_field("value", &map_dtype.value_dtype())?,
                ) else {
                    return Ok(None);
                };

                let entries = Field::new_struct("entries", Fields::from(vec![key, value]), false);
                Field::new(
                    name,
                    DataType::Map(Arc::new(entries), map_dtype.keys_sorted()),
                    nullability.is_nullable(),
                )
            }
            DType::Struct(fields, nullability) => {
                let mut arrow_fields = Vec::with_capacity(fields.nfields());
                for (field, name) in fields.fields().zip(fields.names().iter()) {
                    let Some(field) = self.to_arrow_field(name.as_ref(), &field)? else {
                        return Ok(None);
                    };

                    arrow_fields.push(field);
                }

                Field::new_struct(name, arrow_fields, nullability.is_nullable())
            }
            DType::Extension(ext) if !ext.is::<AnyTemporal>() => {
                let dtype = DType::Extension(ext.clone());
                let exporters = self.exporters(Some(ExportSource::Extension(ext.id())));
                for plugin in exporters.all() {
                    if let Some(field) = plugin.to_arrow_field(name, &dtype, self)? {
                        return Ok(Some(field));
                    }
                }

                // An extension with no plugin that claims it has no Arrow representation.
                return Ok(None);
            }
            DType::Variant(_) => {
                // TODO(Adam): This currently encodes information about parquet-variant
                // at this level. Variant's complexity with being an essentially logical type
                // with multiple physical layout complicates handling this correctly.
                Field::new(
                    name,
                    DataType::Struct(
                        vec![
                            Field::new("metadata", DataType::BinaryView, dtype.is_nullable()),
                            Field::new("value", DataType::BinaryView, dtype.is_nullable()),
                        ]
                        .into(),
                    ),
                    dtype.is_nullable(),
                )
                .with_metadata(
                    [(
                        EXTENSION_TYPE_NAME_KEY.to_string(),
                        "arrow.parquet.variant".to_string(),
                    )]
                    .into(),
                )
            }
            _ => {
                let Some(data_type) = to_data_type_naive(dtype) else {
                    return Ok(None);
                };

                Field::new(name, data_type, dtype.is_nullable())
            }
        };

        Ok(Some(field))
    }

    /// Build the Arrow [`Schema`] for a Vortex top-level [`DType::Struct`], dispatching
    /// extension fields through registered export plugins for inference. Nested
    /// extensions are preserved via [`Self::to_arrow_field`].
    pub fn to_arrow_schema(&self, dtype: &DType) -> VortexResult<Schema> {
        let DType::Struct(struct_dtype, _) = dtype else {
            vortex_bail!("to_arrow_schema requires a top-level struct dtype, got {dtype}");
        };
        let mut fields = Vec::with_capacity(struct_dtype.nfields());
        for (name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
            fields.push(
                self.to_arrow_field(name.as_ref(), &field_dtype)?
                    .ok_or_else(|| no_arrow_type(&field_dtype))?,
            );
        }
        Ok(Schema::new(fields))
    }

    /// Returns the Arrow [`DataType`] that best corresponds to the given Vortex [`DType`],
    /// dispatching [`DType::Extension`]s through registered export plugins.
    ///
    /// Note that a bare [`DataType`] cannot carry `ARROW:extension:name` metadata; use
    /// [`Self::to_arrow_field`] when extension identity must survive the roundtrip.
    pub fn to_arrow_datatype(&self, dtype: &DType) -> VortexResult<DataType> {
        Ok(self
            .to_arrow_field("", dtype)?
            .ok_or_else(|| no_arrow_type(dtype))?
            .data_type()
            .clone())
    }

    /// Build the Vortex [`DType`] for an Arrow [`Field`].
    ///
    /// Plugins registered against the field's Arrow extension name are tried in
    /// registration order; the first plugin to return `Some(dtype)` wins. If none
    /// match (or all return `None`), the builtin `arrow.parquet.variant` extension maps
    /// to [`DType::Variant`], and any other field converts through
    /// [`Self::from_arrow_datatype`] so extension metadata on nested element/struct
    /// fields is preserved.
    #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
    pub fn from_arrow_field(&self, field: &Field) -> VortexResult<DType> {
        if let Some(name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            for plugin in self.importers(&Id::new(name)).iter() {
                if let Some(dtype) = plugin.from_arrow_field(field, self)? {
                    return Ok(dtype);
                }
            }
            // Parquet Variant is understood even without a registered importer plugin.
            if name == "arrow.parquet.variant" {
                return Ok(DType::Variant(field.is_nullable().into()));
            }
        }
        self.from_arrow_datatype(field.data_type(), field.is_nullable().into())
    }

    /// Build the Vortex [`DType`] for an Arrow [`DataType`].
    ///
    /// Recurses into container types ([`DataType::List`] family, [`DataType::FixedSizeList`],
    /// [`DataType::Struct`], [`DataType::Map`], [`DataType::RunEndEncoded`]) via
    /// [`Self::from_arrow_field`] so extension metadata on nested fields dispatches through
    /// registered import plugins. Leaf types use the canonical Arrow → Vortex mapping.
    ///
    /// [`DataType::Dictionary`] is a partial exception: Arrow models its values as a bare
    /// [`DataType`] rather than a [`Field`], so the values themselves cannot carry an extension
    /// name. Fields nested *inside* that data type still do, and are dispatched normally.
    pub fn from_arrow_datatype(
        &self,
        data_type: &DataType,
        nullability: Nullability,
    ) -> VortexResult<DType> {
        Ok(match data_type {
            DataType::List(elem)
            | DataType::LargeList(elem)
            | DataType::ListView(elem)
            | DataType::LargeListView(elem) => {
                DType::List(Arc::new(self.from_arrow_field(elem.as_ref())?), nullability)
            }
            DataType::FixedSizeList(elem, size) => DType::FixedSizeList(
                Arc::new(self.from_arrow_field(elem.as_ref())?),
                *size as u32,
                nullability,
            ),
            DataType::Map(entries, keys_sorted) => {
                vortex_ensure!(
                    !entries.is_nullable(),
                    "Arrow map entries field must be non-nullable"
                );
                let DataType::Struct(fields) = entries.data_type() else {
                    vortex_bail!(
                        "Arrow map entries field must have Struct type, got {:?}",
                        entries.data_type()
                    );
                };
                vortex_ensure_eq!(
                    fields.len(),
                    2,
                    "Arrow map entries struct must contain exactly two fields"
                );
                vortex_ensure!(
                    !fields[0].is_nullable(),
                    "Arrow map key field must be non-nullable"
                );
                DType::map(
                    self.from_arrow_field(fields[0].as_ref())?,
                    self.from_arrow_field(fields[1].as_ref())?,
                    *keys_sorted,
                    nullability,
                )?
            }
            DataType::Struct(fields) => DType::Struct(self.from_arrow_fields(fields)?, nullability),
            DataType::Dictionary(_, value_type) => {
                self.from_arrow_datatype(value_type.as_ref(), nullability)?
            }
            DataType::RunEndEncoded(_, value_field) => {
                self.from_arrow_field(&run_end_values_field(value_field, nullability))?
            }
            _ => from_arrow_data_type(data_type, nullability)?,
        })
    }

    /// Build Vortex [`StructFields`] for Arrow [`Fields`], dispatching each field through
    /// [`Self::from_arrow_field`].
    pub fn from_arrow_fields(&self, fields: &Fields) -> VortexResult<StructFields> {
        fields
            .iter()
            .map(|f| {
                self.from_arrow_field(f)
                    .map(|dt| (FieldName::from(f.name().as_str()), dt))
            })
            .collect::<VortexResult<StructFields>>()
    }

    /// Build the Vortex [`DType`] for an Arrow [`Schema`], dispatching extension fields
    /// through registered import plugins. The result is a top-level non-nullable struct
    /// matching the schema's fields.
    pub fn from_arrow_schema(&self, schema: &Schema) -> VortexResult<DType> {
        Ok(DType::Struct(
            self.from_arrow_fields(schema.fields())?,
            Nullability::NonNullable,
        ))
    }

    /// Decode an Arrow [`RecordBatch`] into a Vortex struct array, dispatching each
    /// extension column through its registered import plugin.
    ///
    /// `schema` is the authoritative Arrow schema used for dispatch — the columns are
    /// consumed positionally. Pass an external schema (rather than relying on
    /// `batch.schema()`) when upstream DataFusion plumbing may have stripped Field-level
    /// extension metadata from the runtime RecordBatch.
    pub fn from_arrow_record_batch(
        &self,
        batch: RecordBatch,
        schema: &Schema,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(batch.num_columns(), schema.fields().len());
        let length = batch.num_rows();
        let names = FieldNames::from_iter(
            schema
                .fields()
                .iter()
                .map(|f| FieldName::from(f.name().as_str())),
        );
        let mut columns = Vec::with_capacity(schema.fields().len());
        for (col, field) in batch.columns().iter().zip(schema.fields().iter()) {
            columns.push(self.from_arrow_array_inner(ArrowArrayRef::clone(col), field)?);
        }
        Ok(StructArray::try_new(names, columns, length, Validity::NonNullable)?.into_array())
    }

    /// Execute a Vortex array into an Arrow array.
    ///
    /// The plugins registered for the array's dtype, then for its encoding, get their turn before
    /// the canonical conversion: first those producing the Arrow extension `target` carries, then
    /// those producing exactly its Arrow type, then those claiming every export from that source.
    /// Only when none of them claims the array does it reach the canonical conversion, which
    /// executes it to a canonical encoding first.
    ///
    /// With `target = None` no Arrow type is requested. The array's own Arrow field is inferred so
    /// an extension dtype still reaches the plugin producing its Arrow extension; everything else
    /// sees no target: plugins naming a type are skipped, plugins naming none choose the type
    /// themselves, and the canonical conversion picks the array's preferred Arrow physical type.
    ///
    /// Uses the default [`ArrowExportOptions`]; see [`exporter`](Self::exporter) to
    /// change them.
    pub fn execute_arrow(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        self.exporter(&ArrowExportOptions::default())
            .execute_arrow(array, target, ctx)
    }

    /// Return an exporter that applies `options` to its exports.
    ///
    /// ```
    /// use arrow_array::Array;
    /// use vortex_array::IntoArray;
    /// use vortex_array::VortexSessionExecute;
    /// use vortex_array::array_session;
    /// use vortex_array::arrays::VarBinViewArray;
    /// use vortex_arrow::ArrowExportOptions;
    /// use vortex_arrow::ArrowSessionExt;
    /// use vortex_arrow::CompactBuffers;
    ///
    /// let session = array_session();
    /// let array = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
    /// let options = ArrowExportOptions::default().with(CompactBuffers(false));
    /// let arrow = session.arrow().exporter(&options).execute_arrow(
    ///     array,
    ///     None,
    ///     &mut session.create_execution_ctx(),
    /// )?;
    /// assert_eq!(arrow.len(), 2);
    /// # Ok::<(), vortex_error::VortexError>(())
    /// ```
    pub fn exporter<'a>(&'a self, options: &'a ArrowExportOptions) -> ArrowExporter<'a> {
        ArrowExporter {
            session: self,
            options,
        }
    }

    /// Decode an Arrow array into a Vortex array.
    ///
    /// `field` describes the Arrow type the array is imported from: pass an Arrow [`Field`] when
    /// one is available, or just the desired nullability (`true` / `false` /
    /// [`Nullability`]) to synthesize an anonymous field from the array's own data type. See
    /// [`IntoArrowField`] for the trade-off between the two.
    ///
    /// Routes through the registered import plugin if `field` carries an Arrow extension
    /// name we recognize, probing each plugin in registration order until one handles the
    /// input or all return [`ArrowImport::Unsupported`]. Otherwise recurses into container
    /// arrays ([`arrow_array::StructArray`], [`arrow_array::GenericListArray`],
    /// [`arrow_array::FixedSizeListArray`], [`arrow_array::GenericListViewArray`]) so
    /// extension fields nested inside containers reach their importers; leaf types fall
    /// through to the canonical Arrow → Vortex array conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if the field (or requested nullability) is non-nullable but the array
    /// physically contains nulls, or if the Arrow data type is unsupported.
    pub fn from_arrow_array<'a>(
        &self,
        array: ArrowArrayRef,
        field: impl IntoArrowField<'a>,
    ) -> VortexResult<ArrayRef> {
        let field = field.into_arrow_field(array.data_type());
        self.from_arrow_array_inner(array, field.as_ref())
    }

    /// [`Self::from_arrow_array`] with the Arrow [`Field`] already resolved: probe the import
    /// plugins registered for the field's extension name, then fall back to the canonical
    /// conversion. Also the recursion point for nested fields, which already have a [`Field`].
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array_inner(
        &self,
        array: ArrowArrayRef,
        field: &Field,
    ) -> VortexResult<ArrayRef> {
        if let Some(extension_name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
            let importers = self.importers(&Id::new(extension_name));
            if !importers.is_empty() {
                let dtype = self.from_arrow_field(field)?;
                let mut current = array;
                for plugin in importers.iter() {
                    match plugin.from_arrow_array(current, field, &dtype, self)? {
                        ArrowImport::Imported(arr) => return Ok(arr),
                        ArrowImport::Unsupported(arr) => current = arr,
                    }
                }
                return self.from_arrow_array_canonical(current.as_ref(), field);
            }
        }
        self.from_arrow_array_canonical(array.as_ref(), field)
    }

    /// Recurse into Arrow container arrays so nested fields with extension metadata reach
    /// their importers, falling through to the canonical conversion for leaf types.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array_canonical(
        &self,
        array: &dyn ArrowArray,
        field: &Field,
    ) -> VortexResult<ArrayRef> {
        use arrow_array::cast::AsArray;

        match field.data_type() {
            DataType::Struct(fields) => {
                let arrow_struct = array.as_struct();
                let names = FieldNames::from_iter(
                    fields.iter().map(|f| FieldName::from(f.name().as_str())),
                );
                let columns = arrow_struct
                    .columns()
                    .iter()
                    .zip(fields.iter())
                    .map(|(col, child_field)| {
                        // Arrow pushes nulls into non-nullable fields; strip before recursing
                        // so Vortex's stricter validity invariants are upheld.
                        let inner = if col.null_count() > 0 && !child_field.is_nullable() {
                            make_array(remove_nulls(col.to_data())?)
                        } else {
                            ArrowArrayRef::clone(col)
                        };
                        self.from_arrow_array_inner(inner, child_field.as_ref())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                let validity = nulls(arrow_struct.nulls(), field.is_nullable())?;
                Ok(
                    StructArray::try_new(names, columns, arrow_struct.len(), validity)?
                        .into_array(),
                )
            }
            DataType::List(elem_field) => {
                let list = array.as_list::<i32>();
                let (offsets, referenced) = trim_offsets(list.offsets());
                let elements = self.from_arrow_array(
                    list.values().slice(referenced.start, referenced.len()),
                    elem_field.as_ref(),
                )?;
                let validity = nulls(list.nulls(), field.is_nullable())?;
                Ok(ListArray::try_new(elements, offsets.into_array(), validity)?.into_array())
            }
            DataType::LargeList(elem_field) => {
                let list = array.as_list::<i64>();
                let (offsets, referenced) = trim_offsets(list.offsets());
                let elements = self.from_arrow_array(
                    list.values().slice(referenced.start, referenced.len()),
                    elem_field.as_ref(),
                )?;
                let validity = nulls(list.nulls(), field.is_nullable())?;
                Ok(ListArray::try_new(elements, offsets.into_array(), validity)?.into_array())
            }
            DataType::FixedSizeList(elem_field, list_size) => {
                let fsl = array.as_fixed_size_list();
                let elements = self.from_arrow_array_inner(
                    ArrowArrayRef::clone(fsl.values()),
                    elem_field.as_ref(),
                )?;
                let validity = nulls(fsl.nulls(), field.is_nullable())?;
                Ok(
                    FixedSizeListArray::try_new(elements, *list_size as u32, validity, fsl.len())?
                        .into_array(),
                )
            }
            DataType::ListView(elem_field) => {
                let list = array.as_list_view::<i32>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let sizes = list.sizes().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable())?;
                Ok(ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array())
            }
            DataType::LargeListView(elem_field) => {
                let list = array.as_list_view::<i64>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let sizes = list.sizes().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable())?;
                Ok(ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array())
            }
            DataType::Map(entries_field, keys_sorted) => {
                let map = array.as_map();
                let entries_array: ArrowArrayRef = Arc::new(map.entries().clone());
                let entries = self.from_arrow_array_inner(entries_array, entries_field.as_ref())?;
                map_from_arrow_parts(
                    entries,
                    map.offsets(),
                    map.nulls(),
                    *keys_sorted,
                    field.is_nullable(),
                )
            }
            DataType::RunEndEncoded(ends_field, values_field) => {
                let values_field = run_end_values_field(values_field, field.is_nullable().into());
                match ends_field.data_type() {
                    DataType::Int16 => self.run_end_from_arrow::<Int16Type>(array, &values_field),
                    DataType::Int32 => self.run_end_from_arrow::<Int32Type>(array, &values_field),
                    DataType::Int64 => self.run_end_from_arrow::<Int64Type>(array, &values_field),
                    ends_dt => vortex_bail!(
                        "Arrow run-end array run ends must be Int16, Int32 or Int64, got {ends_dt}"
                    ),
                }
            }
            DataType::Dictionary(..) => {
                let dict = array.as_any_dictionary();
                // Arrow models dictionary values as a bare `DataType`, so there is no field
                // metadata to carry an extension name for the values themselves. Fields *nested
                // inside* that data type (list elements, struct fields, map entries) do keep
                // their metadata, so importing the values by nullability alone still routes them
                // back through the plugin-aware conversion.
                let values = self
                    .from_arrow_array(ArrowArrayRef::clone(dict.values()), field.is_nullable())?;
                let codes = dict.keys();
                let codes = from_arrow_dyn(codes, codes.is_nullable())?;
                // SAFETY: arrow-rs enforces the dictionary invariants on construction, so the
                // codes are in-bounds for the values.
                Ok(unsafe { DictArray::new_unchecked(codes, values) }.into_array())
            }
            _ => from_arrow_dyn(array, field.is_nullable()),
        }
    }

    /// Decode an Arrow run-end array, recursing into its values so extension metadata on the
    /// values field reaches its importer.
    #[allow(clippy::wrong_self_convention)]
    fn run_end_from_arrow<R: RunEndIndexType>(
        &self,
        array: &dyn ArrowArray,
        values_field: &Field,
    ) -> VortexResult<ArrayRef>
    where
        R::Native: NativePType,
    {
        let run_array = array
            .as_any()
            .downcast_ref::<RunArray<R>>()
            .ok_or_else(|| vortex_err!("expected an Arrow RunArray, got {}", array.data_type()))?;
        let values =
            self.from_arrow_array_inner(ArrowArrayRef::clone(run_array.values()), values_field)?;
        run_end_from_arrow(run_array, values)
    }
}

/// Offer `array` to each plugin in turn, returning the first export any of them claims.
///
/// Plugins that decline hand `array` back, so the caller receives it via
/// [`ArrowExport::Unsupported`] when none of them claims it, wherever in the chain that leaves it.
/// A claimed export is checked against what was asked for, so a misbehaving plugin fails here
/// rather than corrupting an Arrow array further up.
fn probe_exporters(
    plugins: &[ArrowExportVTableRef],
    array: ArrayRef,
    target: Option<&Field>,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowExport> {
    let len = array.len();
    let mut current = array;

    for plugin in plugins {
        trace!(plugin = ?plugin, "probing plugin for converting Vortex array");

        match plugin.execute_arrow(current, target, options, ctx)? {
            ArrowExport::Exported(arrow) => {
                vortex_ensure_eq!(
                    arrow.len(),
                    len,
                    "Arrow array length does not match Vortex array length after conversion by {plugin:?} to {:?}",
                    arrow
                );
                if let Some(target) = target {
                    vortex_ensure!(
                        arrow.data_type() == target.data_type(),
                        "{plugin:?} exported {} but {} was requested",
                        arrow.data_type(),
                        target.data_type()
                    );
                }
                return Ok(ArrowExport::Exported(arrow));
            }
            ArrowExport::Unsupported(array) => current = array,
        }
    }

    Ok(ArrowExport::Unsupported(current))
}

/// The values field of an Arrow [`DataType::RunEndEncoded`], re-stamped with the run-end array's
/// own nullability.
fn run_end_values_field(values_field: &FieldRef, nullability: Nullability) -> Field {
    values_field
        .as_ref()
        .clone()
        .with_nullable(nullability.into())
}

impl SessionVar for ArrowSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// An [`ArrowSession`] bound to [`ArrowExportOptions`], created by [`ArrowSession::exporter`].
#[derive(Clone, Copy, Debug)]
pub struct ArrowExporter<'a> {
    session: &'a ArrowSession,
    options: &'a ArrowExportOptions,
}

impl ArrowExporter<'_> {
    pub(crate) fn options(&self) -> &ArrowExportOptions {
        self.options
    }

    /// Execute a Vortex array into an Arrow array with the bound options.
    ///
    /// Behaves like [`ArrowSession::execute_arrow`]. The options are propagated to nested arrays
    /// and to registered export plugins.
    #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
    pub fn execute_arrow(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let inferred = match target {
            Some(_) => None,
            None => self.session.to_arrow_field("", array.dtype())?,
        };
        let extension_target = target.or(inferred.as_ref());
        let extension = extension_target
            .and_then(|field| field.metadata().get(EXTENSION_TYPE_NAME_KEY))
            .map(|name| Id::new(name));
        let data_type = target.map(Field::data_type);

        let mut current = array;
        for source in ExportSource::of(&current) {
            let exporters = self.session.exporters(source);

            // Extension plugins are the one group handed the inferred field: they need the target
            // to carry the extension name, whether or not the caller asked for one.
            let probes = [
                (
                    extension.and_then(|id| exporters.by_extension.get(&id)),
                    extension_target,
                ),
                (
                    data_type.and_then(|data_type| exporters.by_data_type.get(data_type)),
                    target,
                ),
                (Some(&exporters.any), target),
            ];
            for (plugins, target) in probes {
                let Some(plugins) = plugins.filter(|plugins| !plugins.is_empty()) else {
                    continue;
                };

                trace!(
                    ?source,
                    ?target,
                    plugins = plugins.len(),
                    "probing plugins for converting Vortex array"
                );
                match probe_exporters(plugins, current, target, self.options, ctx)? {
                    ArrowExport::Exported(arrow) => return Ok(arrow),
                    ArrowExport::Unsupported(array) => current = array,
                }
            }
        }

        if extension_target.is_none() {
            // Nothing asked for a type, none could be inferred, and no plugin claimed the array.
            return Err(no_arrow_type(current.dtype()));
        }

        execute_arrow_naive(current, data_type, self, ctx)
    }
}

/// Extension trait for accessing the [`ArrowSession`] on a Vortex session.
pub trait ArrowSessionExt: SessionExt {
    /// Get the Arrow session.
    fn arrow(&self) -> SessionGuard<'_, ArrowSession>;
}

impl<S: SessionExt> ArrowSessionExt for S {
    fn arrow(&self) -> SessionGuard<'_, ArrowSession> {
        self.get::<ArrowSession>()
    }
}

#[cfg(test)]
mod encoding_export_tests;
#[cfg(test)]
mod tests;
