// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plugin layer for moving Arrow extension types in and out of Vortex.
//!
//! Vortex's canonical Arrow conversion (see [`crate::dtype`] and the executor in
//! [`crate::executor`]) handles every non-extension Arrow type and the builtin temporal
//! extensions. The plugins registered here cover the remaining case: **Arrow extension types**.
//!
//! * An [`ArrowExportVTable`] is dispatched purely by the **target Arrow extension Id** —
//!   the plugin is selected when the caller asks for an Arrow [`Field`] carrying matching
//!   `ARROW:extension:name` metadata. The Vortex source dtype/encoding is irrelevant to
//!   dispatch.
//! * An [`ArrowImportVTable`] is dispatched by the **source Arrow extension name** carried
//!   on the incoming [`Field`]. The plugin is responsible for both preserving extension
//!   identity and re-encoding storage if needed (e.g. Arrow `FixedSizeBinary[16]` for UUID
//!   becomes Vortex `FixedSizeList<u8; 16>`).
//!
//! Multiple plugins may register against the same key. They are tried in registration order;
//! each may return [`ArrowExport::Unsupported`] / [`ArrowImport::Unsupported`] to defer to
//! the next.
//!
//! A Vortex extension type that does not have a plugin exports its storage, plus Arrow extension
//! metadata that holds its Vortex ID. [`ArrowSession`] describes this format.

use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::RunArray;
use arrow_array::cast::AsArray;
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
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ExtensionArray;
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
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::session::DTypeSession;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::dtype::session::ExtDTypeRegistry;
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

use crate::ArrowExportOptions;
use crate::IntoVortexArray;
use crate::convert::from_arrow_dyn;
use crate::convert::map_from_arrow_parts;
use crate::convert::nulls;
use crate::convert::remove_nulls;
use crate::convert::trim_map_entries;
use crate::convert::trim_offsets;
use crate::dtype::from_arrow_data_type;
use crate::dtype::to_data_type_naive;
use crate::ext_metadata::decode_arrow_ext_metadata;
use crate::ext_metadata::encode_arrow_ext_metadata;
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

/// Plugin layer for exporting a Vortex array to an Arrow extension type.
///
/// Prefer [`crate::ArrowExtVTable`] for automatic Vortex metadata transport. Both interfaces must
/// satisfy the dtype round-trip contract checked by [`ArrowSession`].
///
/// This is purely an implementation trait, its methods should not be called directly. Instead,
/// use [`ExecuteArrowExt`] to export an array, and the methods on [`ArrowSession`] to convert a
/// dtype.
///
/// [`ExecuteArrowExt`]: crate::ExecuteArrowExt
pub trait ArrowExportVTable: 'static + Send + Sync + Debug {
    /// The Arrow extension ID this plugin produces.
    fn arrow_ext_id(&self) -> Id;

    /// The Vortex array or extension ID this plugin maps from. Used only for inference by
    /// [`ArrowSession::to_arrow_field`] / [`ArrowSession::to_arrow_schema`]; never as a
    /// dispatch key for [`execute_arrow`][Self::execute_arrow].
    fn vortex_id(&self) -> Id;

    /// Build the Arrow [`Field`] this plugin produces for the given Vortex extension
    /// `dtype`. Used during schema inference.
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>>;

    /// Convert a Vortex array into an Arrow array shaped to `target`.
    ///
    /// Returns ownership of `array` via [`ArrowExport::Unsupported`] when the plugin cannot
    /// handle the input. Plugins that export child or storage arrays must pass `options` on to
    /// those exports, through [`ExecuteArrowExt::execute_arrow_with_options`]. The [`ArrowSession`]
    /// of `ctx` is the session of the export, so a plugin that needs one must use it.
    ///
    /// [`ExecuteArrowExt::execute_arrow_with_options`]: crate::ExecuteArrowExt::execute_arrow_with_options
    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport>;
}

/// Plugin layer for importing an Arrow extension-typed array into a Vortex array.
///
/// Prefer [`crate::ArrowExtVTable`] for automatic Vortex metadata transport. Both interfaces must
/// return arrays with the dtype inferred from their fields, as checked by [`ArrowSession`].
///
/// Plugins are dispatched by `arrow_ext_id`. The session peels Arrow dictionaries before it
/// dispatches, so a plugin never sees a [`DataType::Dictionary`] field or array. It sees the value
/// field, with the dictionary field's metadata, and the values array. The session then wraps the
/// plugin's result in a [`DictArray`] with the dictionary's codes.
///
/// Arrow stores dictionary values as a bare [`DataType`], so their extension metadata lives on
/// the outer [`Field`]. Peeling the dictionary lets the importer see the value type described by
/// that metadata. [`DataType::RunEndEncoded`] has a separate values field that can carry its own
/// extension metadata, so its values use the normal nested-field import path.
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
    /// [`Field`] so the plugin is not dispatched again).
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

/// Registry of Arrow exporters, keyed by target Arrow extension [`Id`].
type ArrowExporterRegistry = ArcSwapMap<Id, Arc<[ArrowExportVTableRef]>>;
/// Registry of Arrow exporters, keyed by source Vortex extension [`ExtId`].
type VortexExporterRegistry = ArcSwapMap<ExtId, Arc<[ArrowExportVTableRef]>>;
/// Registry of Arrow importers, keyed by source Arrow extension [`Id`].
type ArrowImporterRegistry = ArcSwapMap<Id, Arc<[ArrowImportVTableRef]>>;

/// Session-scoped registry of Arrow extension plugins.
///
/// All plugins **must** preserve the complete dtype across [`Self::to_arrow_field`] and
/// [`Self::from_arrow_field`]. This includes extension IDs, metadata, storage, and nullability at
/// every nesting level. Exporting an array to that field and importing it with the same field must
/// preserve its dtype and logical values. The owning dtype and Arrow plugins must be registered in
/// the importing session.
///
/// The session checks inferred fields and successful plugin imports, including raw plugins.
/// Explicit export targets may request a different dtype. Unknown Vortex IDs import as storage.
///
/// A plugin converts one Vortex extension type to and from an Arrow extension type. The crate that
/// owns the type registers its plugin, as `vortex-uuid` does. Temporal types need no plugin,
/// because Arrow has native types for them.
///
/// Every [`ArrowSession`] belongs to a [`VortexSession`], which creates it on first use of
/// [`ArrowSessionExt::arrow`]. An export takes no session, because it uses the session of its
/// [`ExecutionCtx`] (see [`ExecuteArrowExt`]). So one export cannot mix the plugins of two sessions.
///
/// An extension type without a plugin exports as its storage, and the Arrow field records the type
/// so that an import can restore it. `ARROW:extension:name` holds the Vortex ID, and
/// `ARROW:extension:metadata` holds the metadata in base64, since Arrow metadata must be UTF-8
/// ([`ext_metadata`] describes the format). DataFusion compares this metadata when it merges file
/// schemas, so [`ExtDTypeRef::serialize_metadata`] must give equal bytes for equal types.
///
/// A caller can also export an array to a field that names its Vortex ID, such as a field from
/// [`Self::to_arrow_storage_field`]. The export then checks that the field describes the type of the
/// array. The field must have the same metadata, and its Arrow type must import as the storage type
/// (or as the type itself, such as an Arrow timestamp). Nested extension types count as their
/// storage types in this check, and the export checks each nested field when it reaches it. Without
/// these checks, values could silently change meaning, such as milliseconds that read back as
/// seconds. For the same reason, an extension array cannot export to a field that names a different
/// known Vortex ID.
///
/// On import, the plugins run first. If none applies, a field that names a known Vortex ID imports
/// as that type over its storage, and a field with any other name imports as plain storage. A
/// Vortex ID is known if its type is registered in the [`DTypeSession`] of the same
/// [`VortexSession`], even after the first use of the Arrow session. The Arrow session keeps the
/// registry of the [`DTypeSession`] that existed at its first use, so it does not see a new
/// [`DTypeSession`] that later replaces it through [`VortexSession::register`].
///
/// [`ExecuteArrowExt`]: crate::ExecuteArrowExt
/// [`ext_metadata`]: crate::ext_metadata
/// [`VortexSession`]: vortex_session::VortexSession
/// [`VortexSession::register`]: vortex_session::VortexSession::register
#[derive(Clone, Debug)]
pub struct ArrowSession {
    exporters: ArrowExporterRegistry,
    exporters_by_vortex: VortexExporterRegistry,
    importers: ArrowImporterRegistry,
    /// The extension dtypes that the session can look up by Vortex ID. This is the registry of a
    /// [`DTypeSession`], so dtypes that are registered there later are also found.
    ext_dtypes: ExtDTypeRegistry,
}

impl ArrowSession {
    /// Create a session without plugins that can look up the extension dtypes of `dtype_session`
    /// by Vortex ID.
    fn new(dtype_session: &DTypeSession) -> Self {
        Self {
            exporters: ArrowExporterRegistry::default(),
            exporters_by_vortex: VortexExporterRegistry::default(),
            importers: ArrowImporterRegistry::default(),
            ext_dtypes: dtype_session.registry().clone(),
        }
    }

    /// Register an [`ArrowExportVTable`] under its target Arrow extension Id (for dispatch)
    /// and its source Vortex extension Id (for schema inference).
    pub fn register_exporter(&self, exporter: ArrowExportVTableRef) {
        self.exporters.push(
            exporter.arrow_ext_id(),
            ArrowExportVTableRef::clone(&exporter),
        );
        self.exporters_by_vortex
            .push(exporter.vortex_id(), exporter);
    }

    /// Register an [`ArrowImportVTable`] under its source Arrow extension name.
    pub fn register_importer(&self, importer: ArrowImportVTableRef) {
        self.importers.push(importer.arrow_ext_id(), importer);
    }

    pub(crate) fn exporters(&self, id: &Id) -> Arc<[ArrowExportVTableRef]> {
        self.exporters.get(id).unwrap_or_else(|| Arc::from([]))
    }

    fn exporters_by_vortex(&self, id: &Id) -> Arc<[ArrowExportVTableRef]> {
        self.exporters_by_vortex
            .get(id)
            .unwrap_or_else(|| Arc::from([]))
    }

    fn importers(&self, id: &Id) -> Arc<[ArrowImportVTableRef]> {
        self.importers.get(id).unwrap_or_else(|| Arc::from([]))
    }

    /// Build the Arrow [`Field`] for a Vortex [`DType`].
    ///
    /// For [`DType::Extension`]s, plugins registered against the extension's `Id`
    /// are tried in registration order; the first plugin to return `Some(field)` wins.
    ///
    /// An extension type without a plugin exports through [`Self::to_arrow_storage_field`].
    pub fn to_arrow_field(&self, name: &str, dtype: &DType) -> VortexResult<Field> {
        // Handle the structural encodings, which may have recursive types
        match dtype {
            DType::List(elem_dtype, nullability) => {
                let elem_field = self.to_arrow_field(Field::LIST_FIELD_DEFAULT_NAME, elem_dtype)?;
                Ok(Field::new_list(name, elem_field, nullability.is_nullable()))
            }
            DType::FixedSizeList(elem_dtype, elem_size, nullability) => {
                let elem_field = self.to_arrow_field(Field::LIST_FIELD_DEFAULT_NAME, elem_dtype)?;
                Ok(Field::new_fixed_size_list(
                    name,
                    elem_field,
                    (*elem_size).try_into()?,
                    nullability.is_nullable(),
                ))
            }
            DType::Map(map_dtype, nullability) => {
                let key = self.to_arrow_field("key", &map_dtype.key_dtype())?;
                let value = self.to_arrow_field("value", &map_dtype.value_dtype())?;
                let entries = Field::new_struct("entries", Fields::from(vec![key, value]), false);
                Ok(Field::new(
                    name,
                    DataType::Map(Arc::new(entries), map_dtype.keys_sorted()),
                    nullability.is_nullable(),
                ))
            }
            DType::Struct(fields, nullability) => {
                let arrow_fields = Fields::from_iter(
                    fields
                        .fields()
                        .zip(fields.names().iter())
                        .map(|(field, name)| self.to_arrow_field(name.as_ref(), &field))
                        .collect::<VortexResult<Vec<_>>>()?,
                );
                Ok(Field::new_struct(
                    name,
                    arrow_fields,
                    nullability.is_nullable(),
                ))
            }
            DType::Extension(ext) if !ext.is::<AnyTemporal>() => {
                match self.to_arrow_field_with_exporter(name, ext)? {
                    Some(field) => Ok(field),
                    None => self.to_arrow_storage_field(name, ext),
                }
            }
            DType::Variant(_) => {
                // TODO(Adam): This currently encodes information about parquet-variant
                // at this level. Variant's complexity with being an essentially logical type
                // with multiple physical layout complicates handling this correctly.
                Ok(Field::new(
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
                ))
            }
            _ => Ok(Field::new(
                name,
                to_data_type_naive(dtype)?,
                dtype.is_nullable(),
            )),
        }
    }

    /// Build the Arrow [`Schema`] for a Vortex top-level [`DType::Struct`], dispatching
    /// extension fields through registered export plugins for inference. Nested
    /// extensions are preserved via [`Self::to_arrow_field`].
    pub fn to_arrow_schema(&self, dtype: &DType) -> VortexResult<Schema> {
        let DType::Struct(struct_dtype, _) = dtype else {
            vortex_bail!("to_arrow_schema requires a top-level struct dtype, got {dtype}");
        };
        let mut fields = Vec::with_capacity(struct_dtype.names().len());
        for (name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
            fields.push(self.to_arrow_field(name.as_ref(), &field_dtype)?);
        }
        Ok(Schema::new(fields))
    }

    /// Returns the Arrow [`DataType`] that best corresponds to the given Vortex [`DType`],
    /// dispatching [`DType::Extension`]s through registered export plugins.
    ///
    /// A [`DataType`] has no field metadata, so this fails for the same types that
    /// [`Self::default_arrow_export_field`] rejects. Use [`Self::to_arrow_field`] to keep the
    /// extension type.
    pub fn to_arrow_datatype(&self, dtype: &DType) -> VortexResult<DataType> {
        Ok(self.default_arrow_export_field(dtype)?.data_type().clone())
    }

    /// Returns the Arrow [`Field`] that an export without a target field uses.
    ///
    /// This is [`Self::to_arrow_field`] with an empty name, except that it fails for a _top-level_
    /// extension type that does not have a plugin. The caller of an export without a target keeps
    /// only the Arrow array or its [`DataType`], not this field. So the extension metadata on the
    /// field would be lost, and with it the extension type. A nested extension type is fine,
    /// because its child [`Field`] keeps the metadata. A top-level extension type with a plugin
    /// still exports, but it loses its extension name, as it did before ([#11286]).
    ///
    /// [#11286]: https://github.com/vortex-data/vortex/issues/11286
    pub fn default_arrow_export_field(&self, dtype: &DType) -> VortexResult<Field> {
        match dtype {
            DType::Extension(ext) if !ext.is::<AnyTemporal>() => {
                self.to_arrow_field_with_exporter("", ext)?.ok_or_else(|| {
                    vortex_err!(
                        "an Arrow export without a target field needs an extension type with an Arrow plugin, got {ext}"
                    )
                })
            }
            _ => self.to_arrow_field("", dtype),
        }
    }

    /// Returns the Arrow [`Field`] from the first exporter that accepts `ext_dtype`, or `None` if
    /// no exporter accepts it.
    fn to_arrow_field_with_exporter(
        &self,
        name: &str,
        ext_dtype: &ExtDTypeRef,
    ) -> VortexResult<Option<Field>> {
        let dtype = DType::Extension(ext_dtype.clone());

        for plugin in self.exporters_by_vortex(&ext_dtype.id()).iter() {
            if let Some(field) = plugin.to_arrow_field(name, &dtype, self)? {
                // Check the completed field through normal routing because another importer can
                // take precedence over the exporter that inferred it.
                vortex_ensure_eq!(
                    self.from_arrow_field(&field)?,
                    dtype,
                    "Arrow field {:?} must preserve its Vortex dtype",
                    name
                );

                return Ok(Some(field));
            }
        }

        Ok(None)
    }

    /// Returns the storage field of `ext_dtype`, with field metadata that holds the extension type
    /// ([`ext_metadata`] describes the format).
    ///
    /// A type with its own Arrow type can also export to this field, for example to export a
    /// timestamp column as `Int64` and import it back as a timestamp.
    ///
    /// Fails if the storage field already has an Arrow extension name.
    ///
    /// [`ext_metadata`]: crate::ext_metadata
    pub fn to_arrow_storage_field(
        &self,
        name: &str,
        ext_dtype: &ExtDTypeRef,
    ) -> VortexResult<Field> {
        let mut field = self.to_arrow_field(name, ext_dtype.storage_dtype())?;
        if let Some(storage_name) = field.extension_type_name() {
            vortex_bail!(
                "the Arrow storage of {ext_dtype} must not be an Arrow extension type, got {storage_name}"
            );
        }

        field
            .metadata_mut()
            .extend(encode_arrow_ext_metadata(ext_dtype)?);

        Ok(field)
    }

    /// Returns the Vortex ID that the extension name `id` resolves to, or `None` if the session does
    /// not know it. The registry can hold an alias, so the result can differ from `id`.
    pub(crate) fn known_vortex_id(&self, id: &Id) -> Option<ExtId> {
        self.ext_dtypes
            .get(id)
            .map(|ext_dtype_plugin| ext_dtype_plugin.id())
    }

    /// Returns the extension type that `field` names by Vortex ID, to wrap around `storage_dtype`.
    ///
    /// Returns `None` if the session does not know that Vortex ID, or if `storage_dtype` already is
    /// that type, such as an Arrow timestamp that names `vortex.timestamp`. Fails if the metadata
    /// of `field` does not fit the type.
    fn ext_dtype_to_wrap(
        &self,
        field: &Field,
        storage_dtype: &DType,
    ) -> VortexResult<Option<ExtDTypeRef>> {
        let Some(extension_name) = field.extension_type_name() else {
            return Ok(None);
        };
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let Some(ext_dtype_plugin) = self.ext_dtypes.get(&Id::new(extension_name)) else {
            return Ok(None);
        };
        let metadata = decode_arrow_ext_metadata(field)?;

        // Wrapping a storage that already is the type would nest the type inside itself.
        let existing_ext_dtype = storage_dtype
            .as_extension_opt()
            .filter(|ext_dtype| ext_dtype.id() == ext_dtype_plugin.id());
        if let Some(existing_ext_dtype) = existing_ext_dtype {
            ensure_ext_metadata(field, &metadata, existing_ext_dtype)?;
            return Ok(None);
        }

        ext_dtype_plugin
            .deserialize(&metadata, storage_dtype.clone())
            .map(Some)
    }

    /// Build the Vortex [`DType`] for an Arrow [`Field`].
    ///
    /// A [`DataType::Dictionary`] field converts as its value field with the same metadata.
    ///
    /// Plugins registered against the field's Arrow extension name are tried in
    /// registration order; the first plugin to return `Some(dtype)` wins. If none
    /// match (or all return `None`), the builtin `arrow.parquet.variant` extension maps
    /// to [`DType::Variant`], and any other field converts through
    /// [`Self::from_arrow_datatype`] so extension metadata on nested element/struct
    /// fields is preserved.
    ///
    /// If the extension name of `field` is a Vortex ID that the session knows, the result is that
    /// extension type over the converted storage (see [`ArrowSession`]).
    #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
    pub fn from_arrow_field(&self, field: &Field) -> VortexResult<DType> {
        if let DataType::Dictionary(_, value_type) = field.data_type() {
            return self
                .from_arrow_field(&field.clone().with_data_type(value_type.as_ref().clone()));
        }

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

        let storage_dtype =
            self.from_arrow_datatype(field.data_type(), field.is_nullable().into())?;

        Ok(match self.ext_dtype_to_wrap(field, &storage_dtype)? {
            Some(ext_dtype) => DType::Extension(ext_dtype),
            None => storage_dtype,
        })
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
    /// If the extension name of `field` is a Vortex ID that the session knows, the import wraps the
    /// converted array in that extension type (see [`ArrowSession`]).
    ///
    /// A dictionary array imports as a [`DictArray`]. Its values import as described above, with
    /// `field`'s metadata.
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

    /// [`Self::from_arrow_array`] with the Arrow [`Field`] already resolved: peel a dictionary,
    /// import through [`Self::from_arrow_array_with_importers`], and wrap the result in the
    /// extension type if the extension name of the field is a Vortex ID that the session knows.
    /// Also the recursion point for nested fields, which already have a [`Field`].
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array_inner(
        &self,
        array: ArrowArrayRef,
        field: &Field,
    ) -> VortexResult<ArrayRef> {
        if let DataType::Dictionary(..) = field.data_type() {
            return self.dictionary_from_arrow(array.as_ref(), field);
        }

        let imported = self.from_arrow_array_with_importers(array, field)?;

        match self.ext_dtype_to_wrap(field, imported.dtype())? {
            Some(ext_dtype) => Ok(ExtensionArray::try_new(ext_dtype, imported)?.into_array()),
            None => Ok(imported),
        }
    }

    /// Probe the import plugins registered for the extension name of `field`, then fall back to
    /// the canonical conversion. Unlike [`Self::from_arrow_array_inner`], this never wraps the
    /// result in the extension type that a Vortex ID names.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array_with_importers(
        &self,
        array: ArrowArrayRef,
        field: &Field,
    ) -> VortexResult<ArrayRef> {
        let mut current = array;

        if let Some(extension_name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
            let importers = self.importers(&Id::new(extension_name));
            if !importers.is_empty() {
                let dtype = self.from_arrow_field(field)?;
                let len = current.len();

                for plugin in importers.iter() {
                    match plugin.from_arrow_array(current, field, &dtype, self)? {
                        // Parents such as lists reuse the Arrow offsets into this array without
                        // checking them again, so a plugin must preserve its length, both in what
                        // it imports and in what it hands back when it declines.
                        ArrowImport::Imported(arr) => {
                            vortex_ensure_eq!(
                                arr.len(),
                                len,
                                "Arrow importer {} returned {} elements for an array of {}",
                                plugin.arrow_ext_id(),
                                arr.len(),
                                len
                            );
                            vortex_ensure_eq!(
                                arr.dtype(),
                                &dtype,
                                "Arrow importer must preserve the dtype inferred from its field"
                            );

                            return Ok(arr);
                        }
                        ArrowImport::Unsupported(arr) => {
                            vortex_ensure_eq!(
                                arr.len(),
                                len,
                                "Arrow importer {} declined and returned {} elements for an array of {}",
                                plugin.arrow_ext_id(),
                                arr.len(),
                                len
                            );
                            current = arr;
                        }
                    }
                }
            }
        }

        self.from_arrow_array_canonical(current.as_ref(), field)
    }

    /// Recurse into Arrow container arrays so nested fields with extension metadata reach
    /// their importers, falling through to the canonical conversion for leaf types.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array_canonical(
        &self,
        array: &dyn ArrowArray,
        field: &Field,
    ) -> VortexResult<ArrayRef> {
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
                // SAFETY: Arrow offsets are non-negative and sorted, and `trim_offsets` rebases
                // them onto the referenced values. The validity has one entry per list.
                Ok(
                    unsafe { ListArray::new_unchecked(elements, offsets.into_array(), validity) }
                        .into_array(),
                )
            }
            DataType::LargeList(elem_field) => {
                let list = array.as_list::<i64>();
                let (offsets, referenced) = trim_offsets(list.offsets());
                let elements = self.from_arrow_array(
                    list.values().slice(referenced.start, referenced.len()),
                    elem_field.as_ref(),
                )?;
                let validity = nulls(list.nulls(), field.is_nullable())?;
                // SAFETY: Arrow offsets are non-negative and sorted, and `trim_offsets` rebases
                // them onto the referenced values. The validity has one entry per list.
                Ok(
                    unsafe { ListArray::new_unchecked(elements, offsets.into_array(), validity) }
                        .into_array(),
                )
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
                // SAFETY: Arrow list views share one integer type for offsets and sizes, and every
                // view lies within the values. The validity has one entry per list.
                Ok(
                    unsafe { ListViewArray::new_unchecked(elements, offsets, sizes, validity) }
                        .into_array(),
                )
            }
            DataType::LargeListView(elem_field) => {
                let list = array.as_list_view::<i64>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let sizes = list.sizes().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable())?;
                // SAFETY: Arrow list views share one integer type for offsets and sizes, and every
                // view lies within the values. The validity has one entry per list.
                Ok(
                    unsafe { ListViewArray::new_unchecked(elements, offsets, sizes, validity) }
                        .into_array(),
                )
            }
            DataType::Map(entries_field, keys_sorted) => {
                let map = array.as_map();
                let (entries, offsets) = trim_map_entries(map);
                // `map_from_arrow_parts` needs the entries as a plain struct, so a Vortex ID on the
                // entries field is ignored. Import plugins for its extension name still run, while
                // `from_arrow_datatype` ignores the entries field completely.
                let entries_array: ArrowArrayRef = Arc::new(entries);
                let entries =
                    self.from_arrow_array_with_importers(entries_array, entries_field.as_ref())?;
                map_from_arrow_parts(
                    entries,
                    &offsets,
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
        run_end_from_arrow(run_array, |values| {
            self.from_arrow_array_inner(values, values_field)
        })
    }

    /// Decode an Arrow dictionary array into a [`DictArray`], importing its values with `field`'s
    /// metadata so that an extension name on the dictionary field reaches its importer.
    fn dictionary_from_arrow(
        &self,
        array: &dyn ArrowArray,
        field: &Field,
    ) -> VortexResult<ArrayRef> {
        let dict = array.as_any_dictionary_opt().ok_or_else(|| {
            vortex_err!(
                "expected an Arrow DictionaryArray, got {}",
                array.data_type()
            )
        })?;

        let values_field = field
            .clone()
            .with_data_type(dict.values().data_type().clone());
        let values =
            self.from_arrow_array_inner(ArrowArrayRef::clone(dict.values()), &values_field)?;

        let codes = dict.keys();
        let codes = from_arrow_dyn(codes, codes.is_nullable())?;

        // SAFETY: arrow-rs enforces the dictionary invariants on construction, so the codes are
        // in-bounds for the values.
        Ok(unsafe { DictArray::new_unchecked(codes, values) }.into_array())
    }
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

/// Fails unless `metadata`, the extension metadata of `field`, is the metadata of `ext_dtype`.
pub(crate) fn ensure_ext_metadata(
    field: &Field,
    metadata: &[u8],
    ext_dtype: &ExtDTypeRef,
) -> VortexResult<()> {
    vortex_ensure!(
        metadata == ext_dtype.serialize_metadata()?,
        "Arrow field {:?} must carry the metadata of {ext_dtype}, got {metadata:?}",
        field.name()
    );

    Ok(())
}

/// Extension trait for accessing the [`ArrowSession`] on a Vortex session.
pub trait ArrowSessionExt: SessionExt {
    /// Get the Arrow session, creating it on first use.
    ///
    /// The new session has no plugins, and it can look up the extension types of
    /// [`DTypeSessionExt::dtypes`] by Vortex ID.
    fn arrow(&self) -> SessionGuard<'_, ArrowSession>;
}

impl<S: SessionExt> ArrowSessionExt for S {
    fn arrow(&self) -> SessionGuard<'_, ArrowSession> {
        self.get_or_insert_with(|| ArrowSession::new(&self.dtypes()))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::DictionaryArray;
    use arrow_array::GenericListArray;
    use arrow_array::GenericListViewArray;
    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_array::MapArray;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_array::types::Int32Type;
    use arrow_buffer::OffsetBuffer;
    use arrow_buffer::ScalarBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::FieldRef;
    use arrow_schema::Fields;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute as _;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::prefix(0, 2)]
    #[case::middle(500, 2)]
    #[case::empty(500, 0)]
    #[case::empty_at_end(1000, 0)]
    fn test_sliced_list_imports_only_its_rows(
        #[values(false, true)] large: bool,
        #[case] start: usize,
        #[case] len: usize,
    ) -> VortexResult<()> {
        let lists = |rows: Range<i64>| -> ArrowArrayRef {
            let values = Arc::new(Int64Array::from_iter_values(
                rows.flat_map(|row| [row, row + 1]),
            ));
            let len = values.len() / 2;
            let field = Arc::new(Field::new("item", DataType::Int64, false));
            if large {
                Arc::new(GenericListArray::<i64>::new(
                    field,
                    OffsetBuffer::from_repeated_length(2, len),
                    values,
                    None,
                ))
            } else {
                Arc::new(GenericListArray::<i32>::new(
                    field,
                    OffsetBuffer::from_repeated_length(2, len),
                    values,
                    None,
                ))
            }
        };
        let session = SESSION.arrow();
        let sliced = session.from_arrow_array(lists(0..1000).slice(start, len), false)?;
        let fresh = session.from_arrow_array(lists(start as i64..(start + len) as i64), false)?;
        assert_eq!(sliced.nbytes(), fresh.nbytes());
        let mut ctx = SESSION.create_execution_ctx();
        let expected = ListArray::new(
            PrimitiveArray::from_iter(
                (start as i64..(start + len) as i64).flat_map(|row| [row, row + 1]),
            )
            .into_array(),
            PrimitiveArray::from_iter((0..=len as u64).map(|row| row * 2)).into_array(),
            Validity::NonNullable,
            &mut ctx,
        );
        assert_arrays_eq!(sliced, expected, &mut ctx);
        assert_arrays_eq!(fresh, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn from_arrow_fields_matches_schema_conversion() -> VortexResult<()> {
        let session = SESSION.arrow();
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8View, true),
        ]);
        let struct_fields = session.from_arrow_fields(&fields)?;
        let schema_dtype = session.from_arrow_schema(&Schema::new(fields))?;
        let expected = StructFields::from_iter([
            ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ("b", DType::Utf8(Nullability::Nullable)),
        ]);
        assert_eq!(struct_fields, expected);
        assert_eq!(
            schema_dtype,
            DType::Struct(struct_fields, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn from_arrow_field_maps_variant_without_importer() -> VortexResult<()> {
        let session = SESSION.arrow();
        let storage = DataType::Struct(
            vec![
                Field::new("metadata", DataType::BinaryView, false),
                Field::new("value", DataType::BinaryView, true),
            ]
            .into(),
        );
        let field = Field::new("v", storage, true).with_metadata(
            [(
                "ARROW:extension:name".to_string(),
                "arrow.parquet.variant".to_string(),
            )]
            .into(),
        );
        assert_eq!(
            session.from_arrow_field(&field)?,
            DType::Variant(Nullability::Nullable)
        );
        Ok(())
    }

    static REJECT_DICTIONARY_ID: CachedId = CachedId::new("test.reject_dictionary");

    /// An importer that fails on an Arrow dictionary and imports anything else as its storage.
    #[derive(Debug)]
    struct DictionaryRejectingImporter;

    impl ArrowImportVTable for DictionaryRejectingImporter {
        fn arrow_ext_id(&self) -> Id {
            *REJECT_DICTIONARY_ID
        }

        fn from_arrow_field(
            &self,
            field: &Field,
            session: &ArrowSession,
        ) -> VortexResult<Option<DType>> {
            vortex_ensure!(
                !matches!(field.data_type(), DataType::Dictionary(..)),
                "an importer must never see a dictionary field, got {}",
                field.data_type()
            );
            session
                .from_arrow_datatype(field.data_type(), field.is_nullable().into())
                .map(Some)
        }

        fn from_arrow_array(
            &self,
            array: ArrowArrayRef,
            field: &Field,
            _dtype: &DType,
            session: &ArrowSession,
        ) -> VortexResult<ArrowImport> {
            vortex_ensure!(
                !matches!(array.data_type(), DataType::Dictionary(..)),
                "an importer must never see a dictionary array, got {}",
                array.data_type()
            );
            session
                .from_arrow_array(array, field.is_nullable())
                .map(ArrowImport::Imported)
        }
    }

    /// An Arrow dictionary imports as a Vortex `Dict` array over the dictionary values, matching
    /// the dtype the schema conversion reports. The session peels the dictionary before any
    /// importer runs, so a dictionary field named for an importer that fails on dictionaries still
    /// imports.
    #[rstest]
    #[case::plain(None)]
    #[case::extension_name(Some(*REJECT_DICTIONARY_ID))]
    fn dictionary_imports_as_dict_encoding(#[case] extension_name: Option<Id>) -> VortexResult<()> {
        let vortex_session = VortexSession::empty();
        let session = vortex_session.arrow();
        session.register_importer(Arc::new(DictionaryRejectingImporter));
        let dict: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1, 0, 1]),
            Arc::new(StringArray::from(vec!["a", "b"])),
        )?);
        let metadata = extension_name
            .map(|id| (EXTENSION_TYPE_NAME_KEY.to_string(), id.to_string()))
            .into_iter()
            .collect();
        let field = Field::new("s", dict.data_type().clone(), false).with_metadata(metadata);

        let dtype = session.from_arrow_field(&field)?;
        assert_eq!(dtype, DType::Utf8(Nullability::NonNullable));

        let array = session.from_arrow_array(dict, &field)?;
        assert!(array.is::<Dict>(), "expected a Dict encoding, got {array}");
        assert_eq!(array.dtype(), &dtype);
        assert_arrays_eq!(
            array,
            VarBinViewArray::from_iter_str(["a", "b", "a", "b"]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    static TRUNCATE_ID: CachedId = CachedId::new("test.truncate");

    /// An importer that drops the last element of every array it is given, then either imports
    /// it or declines and hands it back.
    #[derive(Debug)]
    struct TruncatingImporter {
        decline: bool,
    }

    impl ArrowImportVTable for TruncatingImporter {
        fn arrow_ext_id(&self) -> Id {
            *TRUNCATE_ID
        }

        fn from_arrow_field(
            &self,
            field: &Field,
            session: &ArrowSession,
        ) -> VortexResult<Option<DType>> {
            session
                .from_arrow_datatype(field.data_type(), field.is_nullable().into())
                .map(Some)
        }

        fn from_arrow_array(
            &self,
            array: ArrowArrayRef,
            field: &Field,
            _dtype: &DType,
            session: &ArrowSession,
        ) -> VortexResult<ArrowImport> {
            let truncated = array.slice(0, array.len() - 1);
            if self.decline {
                return Ok(ArrowImport::Unsupported(truncated));
            }
            session
                .from_arrow_array(truncated, field.is_nullable())
                .map(ArrowImport::Imported)
        }
    }

    fn truncated_field(name: &str, data_type: DataType) -> FieldRef {
        Arc::new(
            Field::new(name, data_type, false).with_metadata(
                [(
                    EXTENSION_TYPE_NAME_KEY.to_string(),
                    "test.truncate".to_string(),
                )]
                .into(),
            ),
        )
    }

    fn list_with_truncated_child() -> ArrowArrayRef {
        Arc::new(GenericListArray::<i32>::new(
            truncated_field("item", DataType::Int64),
            OffsetBuffer::new(vec![0, 5].into()),
            Arc::new(Int64Array::from_iter_values(0..5)),
            None,
        ))
    }

    fn list_view_with_truncated_child() -> ArrowArrayRef {
        Arc::new(GenericListViewArray::<i32>::new(
            truncated_field("item", DataType::Int64),
            ScalarBuffer::from(vec![0]),
            ScalarBuffer::from(vec![5]),
            Arc::new(Int64Array::from_iter_values(0..5)),
            None,
        ))
    }

    fn map_with_truncated_child() -> ArrowArrayRef {
        let key_field = Arc::new(Field::new("key", DataType::Int64, false));
        let value_field = Arc::new(Field::new("value", DataType::Int64, false));
        let entries = StructArray::new(
            Fields::from(vec![key_field, value_field]),
            vec![
                Arc::new(Int64Array::from_iter_values(0..5)),
                Arc::new(Int64Array::from_iter_values(5..10)),
            ],
            None,
        );
        Arc::new(MapArray::new(
            truncated_field("entries", entries.data_type().clone()),
            OffsetBuffer::new(vec![0, 5].into()),
            entries,
            None,
            false,
        ))
    }

    #[rstest]
    #[case::list(list_with_truncated_child())]
    #[case::list_view(list_view_with_truncated_child())]
    #[case::map(map_with_truncated_child())]
    fn test_rejects_importer_that_changes_child_length(
        #[case] array: ArrowArrayRef,
        #[values(false, true)] decline: bool,
    ) {
        let vortex_session = VortexSession::empty();
        let session = vortex_session.arrow();
        session.register_importer(Arc::new(TruncatingImporter { decline }));

        let err = session
            .from_arrow_array(array, false)
            .expect_err("a child importer that drops elements must be rejected");
        assert!(
            err.to_string()
                .contains("returned 4 elements for an array of 5"),
            "unexpected error: {err}"
        );
    }
}
