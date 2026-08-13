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
//! * An [`ArrowExportVTable`] converts a Vortex array to Arrow. Its [`ArrowExportKey`] chooses
//!   between the two ways of being dispatched:
//!   * [`ArrowExportKey::ArrowExtension`] — by the **target Arrow extension Id**. The plugin is
//!     selected when the caller asks for an Arrow [`Field`] carrying matching
//!     `ARROW:extension:name` metadata. The Vortex source dtype/encoding is irrelevant to
//!     dispatch.
//!   * [`ArrowExportKey::Encoding`] — by the **source Vortex encoding**, whatever its dtype, and
//!     optionally by the Arrow [`DataType`] asked for.
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
use tracing::debug;
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

use crate::ArrowExportOptions;
use crate::IntoVortexArray;
use crate::convert::from_arrow_dyn;
use crate::convert::map_from_arrow_parts;
use crate::convert::nulls;
use crate::convert::remove_nulls;
use crate::convert::trim_offsets;
use crate::dtype::from_arrow_data_type;
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

/// How an [`ArrowExportVTable`] is dispatched.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArrowExportKey {
    /// Dispatched by the Arrow extension type the caller asked for, whatever the source array is.
    ArrowExtension {
        /// The Arrow extension Id this plugin produces, matched against the
        /// `ARROW:extension:name` of the target [`Field`].
        arrow_ext_id: Id,
        /// The Vortex extension the plugin maps from. Used only for inference by
        /// [`ArrowSession::to_arrow_field`] / [`ArrowSession::to_arrow_schema`], never to
        /// dispatch [`execute_arrow`][ArrowExportVTable::execute_arrow].
        vortex_ext_id: ExtId,
    },
    /// Dispatched by the source Vortex encoding, whatever its dtype.
    Encoding {
        /// The Vortex encoding this plugin exports.
        encoding_id: ArrayId,
        /// The Arrow [`DataType`] this plugin exports to, matched exactly, parameters included: a
        /// plugin producing `Dictionary(UInt8, Utf8)` is not dispatched for a
        /// `Dictionary(UInt16, Utf8)` export.
        ///
        /// `None` claims the encoding for every Arrow type, including an export that requested no
        /// particular type — which such a plugin then chooses for itself. Plugins naming a type
        /// are tried first.
        data_type: Option<DataType>,
    },
}

impl ArrowExportKey {
    /// Dispatch by the Arrow extension `arrow_ext_id`, mapping from the Vortex extension
    /// `vortex_ext_id`.
    pub fn arrow_extension(arrow_ext_id: Id, vortex_ext_id: ExtId) -> Self {
        Self::ArrowExtension {
            arrow_ext_id,
            vortex_ext_id,
        }
    }

    /// Dispatch by the Vortex encoding `encoding_id`, for every Arrow type.
    pub fn encoding(encoding_id: ArrayId) -> Self {
        Self::Encoding {
            encoding_id,
            data_type: None,
        }
    }

    /// Dispatch by the Vortex encoding `encoding_id`, for exports to `data_type` only.
    pub fn encoding_to(encoding_id: ArrayId, data_type: DataType) -> Self {
        Self::Encoding {
            encoding_id,
            data_type: Some(data_type),
        }
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
    /// Only consulted for plugins with an [`ArrowExportKey::ArrowExtension`] key; the default
    /// declines, which is what an encoding-keyed plugin wants.
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        _ = (name, dtype, session);
        Ok(None)
    }

    /// Convert a Vortex array into an Arrow array shaped to `target`.
    ///
    /// `target` is `None` only for a plugin keyed by encoding alone, and only when the export
    /// requested no particular Arrow type: the plugin then picks the type its encoding reaches
    /// most cheaply. Every other plugin is dispatched by something the target carries, so it
    /// always receives one, and must produce exactly its [`DataType`]. Either way the result must
    /// have the same length as `array`.
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

/// Registry of Arrow exporters, keyed by target Arrow extension [`Id`].
type ArrowExporterRegistry = ArcSwapMap<Id, Arc<[ArrowExportVTableRef]>>;
/// Registry of Arrow exporters, keyed by source Vortex extension [`ExtId`].
type VortexExporterRegistry = ArcSwapMap<ExtId, Arc<[ArrowExportVTableRef]>>;
/// Registry of Arrow importers, keyed by source Arrow extension [`Id`].
type ArrowImporterRegistry = ArcSwapMap<Id, Arc<[ArrowImportVTableRef]>>;
/// Registry of Arrow exporters, keyed by source Vortex encoding and the target Arrow [`DataType`]
/// they export to, if they name one.
type EncodingExporterRegistry =
    ArcSwapMap<(ArrayId, Option<DataType>), Arc<[ArrowExportVTableRef]>>;

/// Session-scoped registry of Arrow plugins.
///
/// Exporters are stored in three indices, according to their [`ArrowExportKey`]. Extension-keyed
/// ones go in two: one keyed by Arrow extension Id (used for `execute_arrow` dispatch) and one
/// keyed by Vortex extension Id (used **only** by `to_arrow_field` / `to_arrow_schema` inference,
/// when callers need to translate a Vortex extension `DType` into an Arrow `Field` with no target
/// schema in hand). Encoding-keyed ones go in the third, keyed by the encoding they export and
/// the Arrow type they produce. Importers are keyed by Arrow extension name.
///
/// Extension plugins are registered by the crate that owns the extension type (for example
/// `vortex-uuid`); temporal extensions are handled by the canonical Arrow ↔ Vortex path and do not
/// need plugins.
#[derive(Clone, Debug, Default)]
pub struct ArrowSession {
    exporters: ArrowExporterRegistry,
    exporters_by_vortex: VortexExporterRegistry,
    encoding_exporters: EncodingExporterRegistry,
    importers: ArrowImporterRegistry,
}

impl ArrowSession {
    /// Register an [`ArrowExportVTable`] under its [`ArrowExportKey`].
    ///
    /// An extension-keyed plugin is indexed by its target Arrow extension Id (for dispatch) and
    /// its source Vortex extension Id (for schema inference); an encoding-keyed one by the
    /// encoding it exports and the Arrow type it produces. Plugins sharing a key are tried in
    /// registration order.
    pub fn register_exporter(&self, exporter: ArrowExportVTableRef) {
        match exporter.export_key() {
            ArrowExportKey::ArrowExtension {
                arrow_ext_id,
                vortex_ext_id,
            } => {
                self.exporters
                    .push(arrow_ext_id, ArrowExportVTableRef::clone(&exporter));
                self.exporters_by_vortex.push(vortex_ext_id, exporter);
            }
            ArrowExportKey::Encoding {
                encoding_id,
                data_type,
            } => self
                .encoding_exporters
                .push((encoding_id, data_type), exporter),
        }
    }

    /// Register an [`ArrowImportVTable`] under its source Arrow extension name.
    pub fn register_importer(&self, importer: ArrowImportVTableRef) {
        self.importers.push(importer.arrow_ext_id(), importer);
    }

    fn exporters(&self, id: &Id) -> Arc<[ArrowExportVTableRef]> {
        self.exporters.get(id).unwrap_or_else(|| Arc::from([]))
    }

    /// The plugins registered for `encoding_id` and exactly `data_type`, which is `None` for those
    /// that claim the encoding whatever the Arrow type.
    fn encoding_exporters(
        &self,
        encoding_id: ArrayId,
        data_type: Option<&DataType>,
    ) -> Arc<[ArrowExportVTableRef]> {
        self.encoding_exporters
            .get(&(encoding_id, data_type.cloned()))
            .unwrap_or_else(|| Arc::from([]))
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
                for plugin in self.exporters_by_vortex(&ext.id()).iter() {
                    if let Some(field) =
                        plugin.to_arrow_field(name, &DType::Extension(ext.clone()), self)?
                    {
                        return Ok(field);
                    }
                }
                vortex_bail!("extension type cannot be converted to Arrow without a plugin: {ext}");
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
    /// Note that a bare [`DataType`] cannot carry `ARROW:extension:name` metadata; use
    /// [`Self::to_arrow_field`] when extension identity must survive the roundtrip.
    pub fn to_arrow_datatype(&self, dtype: &DType) -> VortexResult<DataType> {
        Ok(self.to_arrow_field("", dtype)?.data_type().clone())
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
    /// If `target` carries an `ARROW:extension:name`, the plugin registry is probed for one that
    /// can support executing to the target extension type.
    ///
    /// Once no extension identity is left to resolve, the plugins registered for the array's own
    /// encoding get their turn: first those naming the Arrow type being exported to, then those
    /// claiming the encoding whatever the type. Only when none of them claims the array does it
    /// reach the canonical conversion, which executes it to a canonical encoding first.
    ///
    /// With `target = None` no Arrow type is requested: only plugins that named none are
    /// dispatched, and they choose the type themselves. Otherwise the fallback path picks the
    /// array's preferred Arrow physical type and executes directly into that, ignoring extension
    /// types.
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
        // NOTE(aduffy): this looks strange, but we do this to keep target_field as &Field so
        //  we can avoid cloning target when it is provided. It contains a HashMap internally that
        //  can be expensive to copy.
        let arrow_field;
        let target_field = match target {
            Some(field) => field,
            None => {
                arrow_field = match self.session.to_arrow_field("", array.dtype()) {
                    Ok(field) => field,
                    Err(inference_error) => {
                        // An encoding-keyed exporter that chooses its own Arrow type does not need
                        // schema inference to understand the dtype. Give it a chance before
                        // returning the inference error, which is particularly important for an
                        // encoding carrying an extension dtype with no extension exporter.
                        let plugins = self.session.encoding_exporters(array.encoding_id(), None);
                        return match probe_exporters(&plugins, array, None, self.options, ctx)? {
                            ArrowExport::Exported(arrow) => Ok(arrow),
                            ArrowExport::Unsupported(_) => Err(inference_error),
                        };
                    }
                };
                &arrow_field
            }
        };

        if let Some(arrow_ext_name) = target_field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            // There can be multiple plugins that report support for a particular extension type.
            // We try them in order until one of them reports a successful conversion.
            let plugins = self.session.exporters(&Id::new(arrow_ext_name));
            trace!(
                extension_name = arrow_ext_name,
                plugins = plugins.len(),
                "probing extension plugins for converting Vortex array"
            );

            let array =
                match probe_exporters(&plugins, array, Some(target_field), self.options, ctx)? {
                    ArrowExport::Exported(arrow) => return Ok(arrow),
                    ArrowExport::Unsupported(array) => array,
                };

            debug!(
                extension_id = arrow_ext_name,
                data_type = ?target_field.data_type(),
                "unsupported Arrow extension type encountered, falling back to naive execution"
            );

            // The extension target names a concrete Arrow type, so keep it for the rest of the
            // export even though the caller may not have asked for one.
            return self.execute_arrow_by_encoding(array, Some(target_field), ctx);
        }

        self.execute_arrow_by_encoding(array, target, ctx)
    }

    /// Execute a Vortex array into an Arrow array once no Arrow extension identity is left to
    /// resolve: through a plugin registered for the array's encoding if one claims it, otherwise
    /// through the canonical conversion.
    ///
    /// Plugins that named the Arrow type in `target` are tried before those that claim the
    /// encoding whatever the type. With no `target` only the latter are dispatched, and the
    /// canonical conversion is left free to pick the type its own encodings reach most cheaply.
    fn execute_arrow_by_encoding(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let encoding_id = array.encoding_id();
        let data_type = target.map(|target| target.data_type());
        let mut current = array;

        // The plugins that named this export's Arrow type are the more specific ones, so they go
        // first; the `None` pass then covers those claiming the encoding whatever the type. With no
        // type requested, that second pass is the only one.
        for key_data_type in [data_type, None] {
            let plugins = self.session.encoding_exporters(encoding_id, key_data_type);
            if !plugins.is_empty() {
                trace!(
                    encoding = %encoding_id,
                    data_type = ?key_data_type,
                    plugins = plugins.len(),
                    "probing encoding plugins for converting Vortex array"
                );

                match probe_exporters(&plugins, current, target, self.options, ctx)? {
                    ArrowExport::Exported(arrow) => return Ok(arrow),
                    ArrowExport::Unsupported(array) => current = array,
                }
            }
            if key_data_type.is_none() {
                break;
            }
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
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::DictionaryArray;
    use arrow_array::GenericListArray;
    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_array::StringArray;
    use arrow_array::types::Int32Type;
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute as _;
    use vortex_array::array_session;
    use vortex_array::arrays::Dict;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use super::*;

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
        let session = ArrowSession::default();
        let sliced = session.from_arrow_array(lists(0..1000).slice(start, len), false)?;
        let fresh = session.from_arrow_array(lists(start as i64..(start + len) as i64), false)?;
        assert_eq!(sliced.nbytes(), fresh.nbytes());
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(sliced, fresh, &mut ctx);
        Ok(())
    }

    #[test]
    fn from_arrow_fields_matches_schema_conversion() -> VortexResult<()> {
        let session = ArrowSession::default();
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8View, true),
        ]);
        let struct_fields = session.from_arrow_fields(&fields)?;
        let schema_dtype = session.from_arrow_schema(&Schema::new(fields))?;
        assert_eq!(
            schema_dtype,
            DType::Struct(struct_fields, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn from_arrow_field_maps_variant_without_importer() -> VortexResult<()> {
        let session = ArrowSession::default();
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

    /// A plain Arrow dictionary imports as a Vortex `Dict` array over the dictionary values,
    /// matching the dtype the schema conversion reports.
    #[test]
    fn dictionary_imports_as_dict_encoding() -> VortexResult<()> {
        let session = ArrowSession::default();
        let dict: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1, 0, 1]),
            Arc::new(StringArray::from(vec!["a", "b"])),
        )?);
        let field = Field::new("s", dict.data_type().clone(), false);

        let dtype = session.from_arrow_field(&field)?;
        assert_eq!(dtype, DType::Utf8(Nullability::NonNullable));

        let array = session.from_arrow_array(dict, &field)?;
        assert!(array.is::<Dict>(), "expected a Dict encoding, got {array}");
        assert_eq!(array.dtype(), &dtype);
        assert_eq!(array.len(), 4);
        Ok(())
    }
}

#[cfg(test)]
mod encoding_export_tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use arrow_array::Array as ArrowArray;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StringArray;
    use arrow_array::StringViewArray;
    use arrow_array::cast::AsArray;
    use arrow_array::new_null_array;
    use arrow_array::types::Int32Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::EmptyMetadata;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::Extension;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtId;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use super::*;

    const MARKER_INT: i32 = 7;
    const MARKER_STR: &str = "marker";

    /// An extension with no Arrow extension exporter, used to ensure an encoding exporter can
    /// choose a physical Arrow type without schema inference succeeding first.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct UnmappedExtension;

    impl ExtVTable for UnmappedExtension {
        type Metadata = EmptyMetadata;
        type NativeValue<'a> = i32;

        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        fn id(&self) -> ExtId {
            ExtId::new("vortex.arrow.test.unmapped")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(vec![])
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(EmptyMetadata)
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            _storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(0)
        }
    }

    /// A run of marker values, which the canonical conversion could never produce from the arrays
    /// these tests build. That is what tells a test which of the two paths ran.
    fn marker(data_type: &DataType, len: usize) -> VortexResult<ArrowArrayRef> {
        Ok(match data_type {
            DataType::Int32 => Arc::new(Int32Array::from(vec![MARKER_INT; len])),
            DataType::Utf8 => Arc::new(StringArray::from(vec![MARKER_STR; len])),
            DataType::Utf8View => Arc::new(StringViewArray::from(vec![MARKER_STR; len])),
            data_type => vortex_bail!("no marker array for {data_type}"),
        })
    }

    /// Exports every array of one encoding as [`marker`] values.
    #[derive(Debug)]
    struct MarkerExporter {
        key: ArrowExportKey,
        /// The Arrow type it produces when an export requests none.
        data_type: DataType,
    }

    impl MarkerExporter {
        /// Claims `encoding` for exports to `data_type` only.
        fn to_data_type(encoding: ArrayId, data_type: DataType) -> ArrowExportVTableRef {
            Arc::new(Self {
                key: ArrowExportKey::encoding_to(encoding, data_type.clone()),
                data_type,
            })
        }

        /// Claims `encoding` for every export, producing `data_type` when none is requested.
        fn for_encoding(encoding: ArrayId, data_type: DataType) -> ArrowExportVTableRef {
            Arc::new(Self {
                key: ArrowExportKey::encoding(encoding),
                data_type,
            })
        }
    }

    impl ArrowExportVTable for MarkerExporter {
        fn export_key(&self) -> ArrowExportKey {
            self.key.clone()
        }

        fn execute_arrow(
            &self,
            array: ArrayRef,
            target: Option<&Field>,
            _options: &ArrowExportOptions,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrowExport> {
            let data_type = target.map_or(&self.data_type, |target| target.data_type());
            marker(data_type, array.len()).map(ArrowExport::Exported)
        }
    }

    /// Counts how often it is probed and always defers, so a test can assert the plugins ahead of
    /// a claiming one still got their turn.
    #[derive(Debug)]
    struct DeferringExporter {
        key: ArrowExportKey,
        probes: Arc<AtomicUsize>,
    }

    impl DeferringExporter {
        fn registered(key: ArrowExportKey) -> (ArrowExportVTableRef, Arc<AtomicUsize>) {
            let probes = Arc::new(AtomicUsize::new(0));
            let exporter = Arc::new(Self {
                key,
                probes: Arc::clone(&probes),
            });
            (exporter, probes)
        }
    }

    impl ArrowExportVTable for DeferringExporter {
        fn export_key(&self) -> ArrowExportKey {
            self.key.clone()
        }

        fn execute_arrow(
            &self,
            array: ArrayRef,
            _target: Option<&Field>,
            _options: &ArrowExportOptions,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrowExport> {
            self.probes.fetch_add(1, Ordering::Relaxed);
            Ok(ArrowExport::Unsupported(array))
        }
    }

    /// Exports an Arrow type or a length other than the one asked for, to exercise the checks on
    /// what a plugin hands back.
    #[derive(Debug)]
    struct MisbehavingExporter {
        key: ArrowExportKey,
        exported_data_type: DataType,
        len: usize,
    }

    impl ArrowExportVTable for MisbehavingExporter {
        fn export_key(&self) -> ArrowExportKey {
            self.key.clone()
        }

        fn execute_arrow(
            &self,
            _array: ArrayRef,
            _target: Option<&Field>,
            _options: &ArrowExportOptions,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrowExport> {
            Ok(ArrowExport::Exported(new_null_array(
                &self.exported_data_type,
                self.len,
            )))
        }
    }

    fn session() -> VortexSession {
        let session = array_session();
        // Materialize the lazily-created Arrow session up front so registrations and exports share
        // one instance.
        crate::initialize(&session);
        session
    }

    /// `["a", "b", "a"]`, dictionary encoded.
    fn utf8_dict() -> VortexResult<ArrayRef> {
        Ok(DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 0]).into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )?
        .into_array())
    }

    /// `[10, 20, 10]`, dictionary encoded.
    fn primitive_dict() -> VortexResult<ArrayRef> {
        Ok(DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 0]).into_array(),
            PrimitiveArray::from_iter([10i32, 20]).into_array(),
        )?
        .into_array())
    }

    fn export(
        session: &VortexSession,
        array: ArrayRef,
        target: Option<&Field>,
    ) -> VortexResult<ArrowArrayRef> {
        let mut ctx = session.create_execution_ctx();
        session.arrow().execute_arrow(array, target, &mut ctx)
    }

    /// A plugin registered for an encoding claims its arrays whatever their dtype: the same plugin
    /// exports a `Utf8` and a `Primitive` dictionary, neither of which is executed to a canonical
    /// encoding first.
    #[rstest]
    #[case::utf8(utf8_dict(), DataType::Utf8)]
    #[case::primitive(primitive_dict(), DataType::Int32)]
    fn encoding_exporter_claims_its_encoding(
        #[case] array: VortexResult<ArrayRef>,
        #[case] data_type: DataType,
    ) -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), data_type.clone()));

        let field = Field::new("dict", data_type.clone(), false);
        let arrow = export(&session, array?, Some(&field))?;

        assert_eq!(arrow.data_type(), &data_type);
        assert_eq!(arrow.len(), 3);
        assert_marker(&arrow);
        Ok(())
    }

    /// The Arrow type is half the key, so a plugin that named another one is never consulted and
    /// the canonical conversion exports the dictionary's real values.
    #[test]
    fn another_arrow_type_is_a_different_key() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Int32));

        let field = Field::new("dict", DataType::Utf8, false);
        let arrow = export(&session, utf8_dict()?, Some(&field))?;

        assert_eq!(
            string_values(&arrow),
            ["a", "b", "a"],
            "expected the canonical conversion to export the dictionary"
        );
        Ok(())
    }

    /// A plugin that named no Arrow type claims every export of its encoding, and picks the type
    /// itself when none was requested.
    #[rstest]
    #[case::requested(Some(DataType::Utf8), DataType::Utf8)]
    #[case::chosen_by_the_plugin(None, DataType::Utf8View)]
    fn encoding_exporter_without_a_data_type_claims_every_export(
        #[case] requested: Option<DataType>,
        #[case] expected: DataType,
    ) -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::for_encoding(Dict.id(), DataType::Utf8View));

        let field = requested.map(|data_type| Field::new("dict", data_type, false));
        let arrow = export(&session, utf8_dict()?, field.as_ref())?;

        assert_eq!(arrow.data_type(), &expected);
        assert_marker(&arrow);
        Ok(())
    }

    /// An export that requests no Arrow type reaches only the plugins that named none, so one
    /// registered for a specific type stays out of the way.
    #[test]
    fn an_export_of_no_particular_type_skips_type_specific_plugins() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8View));

        let arrow = export(&session, utf8_dict()?, None)?;

        assert_eq!(arrow.data_type(), &DataType::Utf8View);
        let strings = arrow.as_string_view();
        assert_eq!(
            (0..3).map(|i| strings.value(i)).collect::<Vec<_>>(),
            ["a", "b", "a"],
            "expected the canonical conversion to export the dictionary"
        );
        Ok(())
    }

    /// A plugin that chooses its own Arrow type does not need schema inference to understand the
    /// dtype. This lets an encoding export an extension dtype for which no extension plugin exists.
    #[test]
    fn encoding_exporter_can_handle_a_dtype_schema_inference_cannot() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::for_encoding(
                Extension.id(),
                DataType::Int32,
            ));

        let storage = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let array = ExtensionArray::try_new_from_vtable(UnmappedExtension, EmptyMetadata, storage)?
            .into_array();
        let arrow = export(&session, array, None)?;

        assert_eq!(arrow.data_type(), &DataType::Int32);
        assert_marker(&arrow);
        Ok(())
    }

    /// Eager initialization must not replace an Arrow session that encoding initializers already
    /// materialized and populated.
    #[test]
    fn initialize_preserves_lazily_registered_exporters() -> VortexResult<()> {
        let session = array_session();
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

        crate::initialize(&session);

        let field = Field::new("dict", DataType::Utf8, false);
        let arrow = export(&session, utf8_dict()?, Some(&field))?;
        assert_marker(&arrow);
        Ok(())
    }

    /// The plugins that named the Arrow type being exported to are more specific, so they are
    /// tried before those that claim the encoding whatever the type.
    #[test]
    fn type_specific_plugins_are_tried_first() -> VortexResult<()> {
        let session = session();
        let (deferring, probes) =
            DeferringExporter::registered(ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8));
        session.arrow().register_exporter(deferring);
        session
            .arrow()
            .register_exporter(MarkerExporter::for_encoding(Dict.id(), DataType::Utf8));

        let field = Field::new("dict", DataType::Utf8, false);
        let arrow = export(&session, utf8_dict()?, Some(&field))?;

        assert_eq!(probes.load(Ordering::Relaxed), 1);
        assert_marker(&arrow);
        Ok(())
    }

    /// A plugin that defers hands the array on, and with no other plugin to claim it the canonical
    /// conversion exports the dictionary's real values.
    #[test]
    fn deferring_encoding_exporter_falls_through() -> VortexResult<()> {
        let session = session();
        let (deferring, probes) =
            DeferringExporter::registered(ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8));
        session.arrow().register_exporter(deferring);

        let field = Field::new("dict", DataType::Utf8, false);
        let arrow = export(&session, utf8_dict()?, Some(&field))?;

        assert_eq!(probes.load(Ordering::Relaxed), 1);
        assert_eq!(string_values(&arrow), ["a", "b", "a"]);
        Ok(())
    }

    /// Plugins registered under one key are tried in registration order.
    #[test]
    fn encoding_exporters_are_tried_in_registration_order() -> VortexResult<()> {
        let session = session();
        let (deferring, probes) =
            DeferringExporter::registered(ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8));
        session.arrow().register_exporter(deferring);
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

        let field = Field::new("dict", DataType::Utf8, false);
        let arrow = export(&session, utf8_dict()?, Some(&field))?;

        assert_eq!(probes.load(Ordering::Relaxed), 1);
        assert_marker(&arrow);
        Ok(())
    }

    /// Columns nested inside a struct route back through the session, so their encodings reach
    /// their plugins too.
    #[test]
    fn encoding_exporter_claims_a_nested_column() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

        let array = StructArray::try_new(
            FieldNames::from(["dict"]),
            vec![utf8_dict()?],
            3,
            Validity::NonNullable,
        )?
        .into_array();

        let field = Field::new(
            "row",
            DataType::Struct(Fields::from(vec![Field::new(
                "dict",
                DataType::Utf8,
                false,
            )])),
            false,
        );
        let arrow = export(&session, array, Some(&field))?;

        assert_marker(arrow.as_struct().column(0));
        Ok(())
    }

    /// An encoding with no registered plugin is exported by the canonical conversion, as before.
    #[test]
    fn unregistered_encodings_use_the_canonical_conversion() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

        let field = Field::new("s", DataType::Utf8, false);
        let arrow = export(
            &session,
            VarBinViewArray::from_iter_str(["a", "b", "a"]).into_array(),
            Some(&field),
        )?;

        assert_eq!(string_values(&arrow), ["a", "b", "a"]);
        Ok(())
    }

    /// A plugin that drops or invents rows is rejected rather than silently corrupting the export.
    #[test]
    fn encoding_exporter_must_preserve_length() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(Arc::new(MisbehavingExporter {
                key: ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8),
                exported_data_type: DataType::Utf8,
                len: 2,
            }));

        let field = Field::new("dict", DataType::Utf8, true);
        let err = export(&session, utf8_dict()?, Some(&field))
            .expect_err("expected a length mismatch to be rejected");

        assert!(
            err.to_string().contains("length does not match"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    /// A plugin that exports an Arrow type other than the one requested is rejected too: its
    /// caller asked for a specific type and would otherwise get another.
    #[test]
    fn encoding_exporter_must_honor_the_requested_type() -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(Arc::new(MisbehavingExporter {
                key: ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8),
                exported_data_type: DataType::Int32,
                len: 3,
            }));

        let field = Field::new("dict", DataType::Utf8, true);
        let err = export(&session, utf8_dict()?, Some(&field))
            .expect_err("expected the wrong Arrow type to be rejected");

        assert!(
            err.to_string().contains("but Utf8 was requested"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    fn string_values(arrow: &ArrowArrayRef) -> Vec<&str> {
        let strings = arrow.as_string::<i32>();
        (0..strings.len()).map(|i| strings.value(i)).collect()
    }

    fn assert_marker(arrow: &ArrowArrayRef) {
        match arrow.data_type() {
            DataType::Utf8 => {
                assert!(
                    string_values(arrow)
                        .iter()
                        .all(|value| *value == MARKER_STR),
                    "expected the exporter's marker values, got {arrow:?}"
                );
            }
            DataType::Utf8View => {
                let strings = arrow.as_string_view();
                assert!(
                    (0..strings.len()).all(|i| strings.value(i) == MARKER_STR),
                    "expected the exporter's marker values, got {arrow:?}"
                );
            }
            DataType::Int32 => {
                let ints = arrow.as_primitive::<Int32Type>();
                assert!(
                    ints.values().iter().all(|value| *value == MARKER_INT),
                    "expected the exporter's marker values, got {arrow:?}"
                );
            }
            data_type => panic!("unexpected marker type {data_type}"),
        }
    }
}
