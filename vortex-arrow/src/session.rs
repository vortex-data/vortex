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

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::Array as _;
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
use arrow_schema::extension::ExtensionType;
use tracing::debug;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
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
use vortex_array::extension::uuid::Uuid;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::FromArrowArray;
use crate::IntoVortexArray;
use crate::convert::map_from_arrow_parts;
use crate::convert::nulls;
use crate::convert::remove_nulls;
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

/// Plugin layer for exporting a Vortex array to an Arrow extension type.
///
/// This is purely an implementation trait, its methods should not be called directly. Instead,
/// use the methods on [`ArrowSession`].
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
    /// handle the input.
    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
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
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
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
/// Exporters are stored in two indices: one keyed by Arrow extension Id (used for
/// `execute_arrow` dispatch) and one keyed by Vortex extension Id (used **only** by
/// `to_arrow_field` / `to_arrow_schema` inference, when callers need to translate a Vortex
/// extension `DType` into an Arrow `Field` with no target schema in hand). Importers are
/// keyed by Arrow extension name. The default session pre-registers the builtin UUID
/// plugin; temporal extensions are handled by the canonical Arrow ↔ Vortex path and do not
/// need plugins.
#[derive(Clone, Debug)]
pub struct ArrowSession {
    exporters: ArrowExporterRegistry,
    exporters_by_vortex: VortexExporterRegistry,
    importers: ArrowImporterRegistry,
}

impl Default for ArrowSession {
    fn default() -> Self {
        let session = Self {
            exporters: ArrowExporterRegistry::default(),
            exporters_by_vortex: VortexExporterRegistry::default(),
            importers: ArrowImporterRegistry::default(),
        };

        session.register_exporter(Arc::new(Uuid));
        session.register_importer(Arc::new(Uuid));

        session
    }
}

impl ArrowSession {
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

    fn exporters(&self, id: &Id) -> Arc<[ArrowExportVTableRef]> {
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
    /// [`DataType::Dictionary`] is the one exception: Arrow models its values as a bare
    /// [`DataType`] rather than a [`Field`], so dictionary values cannot carry extension
    /// metadata and are converted with the naive mapping.
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
                vortex_ensure!(
                    fields.len() == 2,
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
        vortex_ensure!(
            batch.num_columns() == schema.fields().len(),
            "RecordBatch has {} columns but schema has {} fields",
            batch.num_columns(),
            schema.fields().len()
        );
        let length = batch.num_rows();
        let names = FieldNames::from_iter(
            schema
                .fields()
                .iter()
                .map(|f| FieldName::from(f.name().as_str())),
        );
        let mut columns = Vec::with_capacity(schema.fields().len());
        for (col, field) in batch.columns().iter().zip(schema.fields().iter()) {
            columns.push(self.from_arrow_array(ArrowArrayRef::clone(col), field)?);
        }
        Ok(StructArray::try_new(names, columns, length, Validity::NonNullable)?.into_array())
    }

    /// Execute a Vortex array into an Arrow array.
    ///
    /// If `target` carries an `ARROW:extension:name`, the plugin registry is probed for one that
    /// can support executing to the target extension type.
    ///
    /// With `target = None` the fallback path picks the array's preferred Arrow physical type
    /// and executes directly into that, ignoring extension types.
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
                let session = ctx.session().clone();
                arrow_field = session.arrow().to_arrow_field("", array.dtype())?;
                &arrow_field
            }
        };

        if let Some(arrow_ext_name) = target_field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            // There can be multiple plugins that report support for a particular extension type.
            // We try them in order until one of them reports a successful conversion.
            let len = array.len();
            let mut current = array;

            for plugin in self.exporters(&Id::new(arrow_ext_name)).iter() {
                trace!(
                    plugin = ?plugin,
                    extension_name = arrow_ext_name,
                    "probing plugin for converting Arrow array"
                );

                match plugin.execute_arrow(current, target_field, ctx)? {
                    ArrowExport::Exported(arrow) => {
                        vortex_ensure!(
                            arrow.len() == len,
                            "Arrow array length does not match Vortex array length after conversion to {:?}",
                            arrow
                        );
                        return Ok(arrow);
                    }
                    ArrowExport::Unsupported(array) => current = array,
                }
            }

            debug!(
                extension_id = arrow_ext_name,
                data_type = ?target_field.data_type(),
                "unsupported Arrow extension type encountered, falling back to naive execution"
            );

            return execute_arrow_naive(current, Some(target_field.data_type()), ctx);
        }

        execute_arrow_naive(array, target.map(|field| field.data_type()), ctx)
    }

    /// Decode an Arrow array into a Vortex array.
    ///
    /// Routes through the registered import plugin if `field` carries an Arrow extension
    /// name we recognize, probing each plugin in registration order until one handles the
    /// input or all return [`ArrowImport::Unsupported`]. Otherwise recurses into container
    /// arrays ([`arrow_array::StructArray`], [`arrow_array::GenericListArray`],
    /// [`arrow_array::FixedSizeListArray`], [`arrow_array::GenericListViewArray`]) so
    /// extension fields nested inside containers reach their importers; leaf types fall
    /// through to the canonical Arrow → Vortex array conversion.
    pub fn from_arrow_array(&self, array: ArrowArrayRef, field: &Field) -> VortexResult<ArrayRef> {
        if let Some(extension_name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
            let importers = self.importers(&Id::new(extension_name));
            if !importers.is_empty() {
                let dtype = self.from_arrow_field(field)?;
                let mut current = array;
                for plugin in importers.iter() {
                    match plugin.from_arrow_array(current, field, &dtype)? {
                        ArrowImport::Imported(arr) => return Ok(arr),
                        ArrowImport::Unsupported(arr) => current = arr,
                    }
                }
                return ArrayRef::from_arrow(current.as_ref(), field.is_nullable());
            }
        }
        self.from_arrow_array_canonical(array, field)
    }

    /// Recurse into Arrow container arrays so nested fields with extension metadata reach
    /// their importers, falling through to [`ArrayRef::from_arrow`] for leaf types.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array_canonical(
        &self,
        array: ArrowArrayRef,
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
                        self.from_arrow_array(inner, child_field.as_ref())
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
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable())?;
                Ok(ListArray::try_new(elements, offsets, validity)?.into_array())
            }
            DataType::LargeList(elem_field) => {
                let list = array.as_list::<i64>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable())?;
                Ok(ListArray::try_new(elements, offsets, validity)?.into_array())
            }
            DataType::FixedSizeList(elem_field, list_size) => {
                let fsl = array.as_fixed_size_list();
                let elements =
                    self.from_arrow_array(ArrowArrayRef::clone(fsl.values()), elem_field.as_ref())?;
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
                let entries = self.from_arrow_array(entries_array, entries_field.as_ref())?;
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
                    DataType::Int16 => self.run_end_from_arrow::<Int16Type>(&array, &values_field),
                    DataType::Int32 => self.run_end_from_arrow::<Int32Type>(&array, &values_field),
                    DataType::Int64 => self.run_end_from_arrow::<Int64Type>(&array, &values_field),
                    ends_dt => vortex_bail!(
                        "Arrow run-end array run ends must be Int16, Int32 or Int64, got {ends_dt}"
                    ),
                }
            }
            _ => ArrayRef::from_arrow(array.as_ref(), field.is_nullable()),
        }
    }

    /// Decode an Arrow run-end array, recursing into its values so extension metadata on the
    /// values field reaches its importer.
    #[allow(clippy::wrong_self_convention)]
    fn run_end_from_arrow<R: RunEndIndexType>(
        &self,
        array: &ArrowArrayRef,
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
            self.from_arrow_array(ArrowArrayRef::clone(run_array.values()), values_field)?;
        run_end_from_arrow(run_array, values)
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

// NOTE(aduffy): We should remove this once we bump Arrow to 0.59.0. This is replicating the
//  `Field::has_valid_extension_type` method on Arrow added in 58.2.0, we polyfill it here so that
//  this crate can build with minimal-versions declared.
pub(crate) fn has_valid_extension_type<E: ExtensionType>(field: &Field) -> bool {
    if field.extension_type_name() != Some(E::NAME) {
        return false;
    }

    E::try_new_from_field_metadata(field.data_type(), field.metadata()).is_ok()
}

impl SessionVar for ArrowSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
    use std::sync::Arc;

    use arrow_array::FixedSizeBinaryArray;
    use arrow_array::Int32Array;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::Uuid as ArrowUuid;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::extension::uuid::Uuid;
    use vortex_array::extension::uuid::UuidMetadata;
    use vortex_error::VortexResult;

    use super::*;

    fn uuid_dtype(nullable: bool) -> DType {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            nullable.into(),
        );
        DType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage)
                .expect("uuid ext dtype")
                .erased(),
        )
    }

    #[test]
    fn to_arrow_field_top_level_uuid_carries_extension_metadata() -> VortexResult<()> {
        let session = ArrowSession::default();
        let field = session.to_arrow_field("id", &uuid_dtype(false))?;
        assert!(has_valid_extension_type::<ArrowUuid>(&field));
        Ok(())
    }

    #[test]
    fn to_arrow_field_struct_with_nested_uuid_preserves_metadata() -> VortexResult<()> {
        let session = ArrowSession::default();
        let dtype = DType::Struct(
            StructFields::from_iter([(FieldName::from("id"), uuid_dtype(false))]),
            Nullability::NonNullable,
        );
        let field = session.to_arrow_field("row", &dtype)?;
        let DataType::Struct(inner) = field.data_type() else {
            panic!("expected Struct, got {:?}", field.data_type());
        };
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].data_type(), &DataType::FixedSizeBinary(16));
        assert!(has_valid_extension_type::<ArrowUuid>(&inner[0]));
        Ok(())
    }

    #[test]
    fn to_arrow_field_list_of_uuid_preserves_metadata() -> VortexResult<()> {
        let session = ArrowSession::default();
        let dtype = DType::List(Arc::new(uuid_dtype(true)), Nullability::NonNullable);
        let field = session.to_arrow_field("ids", &dtype)?;
        let DataType::List(elem) = field.data_type() else {
            panic!("expected List, got {:?}", field.data_type());
        };
        assert!(has_valid_extension_type::<ArrowUuid>(elem));
        Ok(())
    }

    #[test]
    fn to_arrow_field_fixed_size_list_of_uuid_preserves_metadata() -> VortexResult<()> {
        let session = ArrowSession::default();
        let dtype = DType::FixedSizeList(Arc::new(uuid_dtype(false)), 3, Nullability::NonNullable);
        let field = session.to_arrow_field("triple", &dtype)?;
        let DataType::FixedSizeList(elem, size) = field.data_type() else {
            panic!("expected FixedSizeList, got {:?}", field.data_type());
        };
        assert_eq!(*size, 3);
        assert!(has_valid_extension_type::<ArrowUuid>(elem));
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_map_uuid_fields() -> VortexResult<()> {
        let session = ArrowSession::default();
        let map = DType::map(
            uuid_dtype(false),
            uuid_dtype(true),
            true,
            Nullability::Nullable,
        )?;
        let dtype = DType::Struct(
            StructFields::from_iter([(FieldName::from("ids"), map)]),
            Nullability::NonNullable,
        );

        let schema = session.to_arrow_schema(&dtype)?;
        let field = schema.field(0);
        let DataType::Map(entries, keys_sorted) = field.data_type() else {
            panic!("expected Map, got {:?}", field.data_type());
        };
        assert!(*keys_sorted);
        assert_eq!(entries.name(), "entries");
        assert!(!entries.is_nullable());
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected map entries struct, got {:?}", entries.data_type());
        };
        assert!(has_valid_extension_type::<ArrowUuid>(&fields[0]));
        assert!(has_valid_extension_type::<ArrowUuid>(&fields[1]));
        assert!(!fields[0].is_nullable());
        assert!(fields[1].is_nullable());

        assert_eq!(session.from_arrow_schema(&schema)?, dtype);
        Ok(())
    }

    #[test]
    fn to_arrow_schema_struct_of_struct_uuid() -> VortexResult<()> {
        let session = ArrowSession::default();
        let inner = DType::Struct(
            StructFields::from_iter([(FieldName::from("id"), uuid_dtype(true))]),
            Nullability::NonNullable,
        );
        let outer = DType::Struct(
            StructFields::from_iter([(FieldName::from("payload"), inner)]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&outer)?;
        let payload = schema.field(0);
        let DataType::Struct(inner_fields) = payload.data_type() else {
            panic!("expected Struct, got {:?}", payload.data_type());
        };
        assert!(has_valid_extension_type::<ArrowUuid>(&inner_fields[0]));
        Ok(())
    }

    #[test]
    fn from_arrow_field_recurses_into_nested_uuid() -> VortexResult<()> {
        let session = ArrowSession::default();
        let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
        elem.try_with_extension_type(ArrowUuid)?;
        let outer = Field::new("ids", DataType::List(Arc::new(elem)), false);

        let dtype = session.from_arrow_field(&outer)?;
        let DType::List(inner_dt, _) = dtype else {
            panic!("expected List dtype, got {dtype}");
        };
        assert!(
            matches!(inner_dt.as_ref(), DType::Extension(ext) if ext.id() == Uuid.id()),
            "expected Uuid extension element, got {inner_dt}",
        );
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_nested_uuid() -> VortexResult<()> {
        let session = ArrowSession::default();
        let dtype = DType::Struct(
            StructFields::from_iter([
                (FieldName::from("id"), uuid_dtype(false)),
                (
                    FieldName::from("ids"),
                    DType::List(Arc::new(uuid_dtype(true)), Nullability::NonNullable),
                ),
            ]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&dtype)?;
        let roundtripped = session.from_arrow_schema(&schema)?;
        assert_eq!(roundtripped, dtype);
        Ok(())
    }

    #[test]
    fn to_arrow_datatype_dispatches_plugins() -> VortexResult<()> {
        let session = ArrowSession::default();
        assert_eq!(
            session.to_arrow_datatype(&uuid_dtype(false))?,
            DataType::FixedSizeBinary(16)
        );
        assert_eq!(
            session.to_arrow_datatype(&DType::Utf8(Nullability::Nullable))?,
            DataType::Utf8View
        );
        Ok(())
    }

    #[test]
    fn from_arrow_datatype_recurses_into_nested_extension_fields() -> VortexResult<()> {
        let session = ArrowSession::default();
        let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
        elem.try_with_extension_type(ArrowUuid)?;
        let data_type = DataType::List(Arc::new(elem));

        let dtype = session.from_arrow_datatype(&data_type, Nullability::Nullable)?;
        let DType::List(inner_dt, Nullability::Nullable) = dtype else {
            panic!("expected nullable List dtype, got {dtype}");
        };
        assert!(
            matches!(inner_dt.as_ref(), DType::Extension(ext) if ext.id() == Uuid.id()),
            "expected Uuid extension element, got {inner_dt}",
        );
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

    #[test]
    fn execute_arrow_target_none_preserves_top_level_uuid_metadata() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
        field.try_with_extension_type(ArrowUuid)?;
        let arrow_array: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?);

        let vortex_array = session.from_arrow_array(arrow_array, &field)?;

        let vortex_ext = vortex_array.dtype().as_extension();
        assert!(vortex_ext.is::<Uuid>());

        let exported = session.execute_arrow(vortex_array, None, &mut ctx)?;
        assert_eq!(exported.data_type(), &DataType::FixedSizeBinary(16));
        let fsb = exported.as_fixed_size_binary();
        assert_eq!(fsb.len(), 2);
        assert_eq!(fsb.value(0), b"0123456789abcdef");
        assert_eq!(fsb.value(1), b"fedcba9876543210");
        Ok(())
    }

    /// An Arrow run-end array whose values field carries extension metadata must import as that
    /// extension, through both the dtype and the array conversion.
    #[test]
    fn run_end_recurses_into_extension_values() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let mut values_field = Field::new("values", DataType::FixedSizeBinary(16), false);
        values_field.try_with_extension_type(ArrowUuid)?;
        let field = Field::new(
            "id",
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                Arc::new(values_field),
            ),
            false,
        );

        let dtype = session.from_arrow_field(&field)?;
        assert!(
            dtype.as_extension().is::<Uuid>(),
            "expected a Uuid extension dtype, got {dtype}"
        );

        let values = FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?;
        let run_array: ArrowArrayRef = Arc::new(RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![2i32, 5]),
            &values,
        )?);

        let vortex_array = session.from_arrow_array(run_array, &field)?;
        assert_eq!(vortex_array.len(), 5);
        // The array conversion must agree with the dtype conversion.
        assert_eq!(vortex_array.dtype(), &dtype);

        // And the values must round-trip back out through the export plugin.
        let exported = session.execute_arrow(vortex_array, Some(&field), &mut ctx)?;
        assert_eq!(exported.len(), 5);
        let ree = exported
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .ok_or_else(|| {
                vortex_err!(
                    "expected an Int32 run-end array, got {}",
                    exported.data_type()
                )
            })?;
        let values = ree.values().as_fixed_size_binary();
        assert_eq!(values.value(0), b"0123456789abcdef");
        assert_eq!(values.value(1), b"fedcba9876543210");
        Ok(())
    }
}
