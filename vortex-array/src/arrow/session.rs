// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plugin layer for moving Arrow extension types in and out of Vortex.
//!
//! Vortex's canonical Arrow conversion (see [`crate::dtype::arrow`] and the executor in
//! [`crate::arrow::executor`]) handles every non-extension Arrow type and the builtin temporal
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

use arc_swap::ArcSwap;
use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Id;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::arrow::FromArrowArray;
use crate::arrow::executor::canonical_execute_arrow;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::dtype::arrow::FromArrowType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::extension::uuid::Uuid;
use crate::validity::Validity;

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
/// Plugins are dispatched purely by [`arrow_ext_id`][Self::arrow_ext_id]: when the caller
/// asks the session to export to an Arrow [`Field`] whose `ARROW:extension:name` matches,
/// this plugin's [`execute_arrow`][Self::execute_arrow] is invoked.
///
/// [`vortex_ext_id`][Self::vortex_ext_id] is **not** used for dispatch. It is consulted only
/// by [`ArrowSession::to_arrow_field`] / [`ArrowSession::to_arrow_schema`] so that a Vortex
/// extension `DType` can be turned into a proper Arrow [`Field`] (with the right
/// `ARROW:extension:name` metadata) when no target schema is supplied — for example when
/// DataFusion is asking Vortex to describe a file's schema.
pub trait ArrowExportVTable: 'static + Send + Sync + Debug {
    /// The Arrow extension Id this plugin produces.
    fn arrow_ext_id(&self) -> Id;

    /// The Vortex extension Id this plugin maps from. Used only for inference by
    /// [`ArrowSession::to_arrow_field`] / [`ArrowSession::to_arrow_schema`]; never as a
    /// dispatch key for [`execute_arrow`][Self::execute_arrow].
    fn vortex_ext_id(&self) -> ExtId;

    /// Build the Arrow [`Field`] this plugin produces for the given Vortex extension
    /// `dtype`. Used during schema inference.
    fn to_arrow_field(&self, name: &str, dtype: &ExtDTypeRef) -> VortexResult<Field>;

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

/// Plugin layer for importing an Arrow extension-typed array into a Vortex extension array.
///
/// Plugins are dispatched by [`arrow_ext_name`][Self::arrow_ext_name]: when the session sees
/// an Arrow [`Field`] whose `ARROW:extension:name` matches, this plugin's
/// [`from_arrow_array`][Self::from_arrow_array] is invoked.
pub trait ArrowImportVTable: 'static + Send + Sync + Debug {
    /// The Arrow extension name this plugin handles.
    fn arrow_ext_id(&self) -> Id;

    /// Build the Vortex [`DType`] that corresponds to `field` (which carries this plugin's
    /// Arrow extension metadata).
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_field(&self, field: &Field) -> VortexResult<DType>;

    /// Convert an Arrow array into a Vortex extension array of `dtype`.
    ///
    /// Returns ownership of `array` via [`ArrowImport::Unsupported`] when the plugin cannot
    /// handle the input.
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        dtype: &ExtDTypeRef,
    ) -> VortexResult<ArrowImport>;
}

pub type ArrowExportVTableRef = Arc<dyn ArrowExportVTable>;
pub type ArrowImportVTableRef = Arc<dyn ArrowImportVTable>;

type ExportMap = HashMap<Id, Vec<ArrowExportVTableRef>>;
type ImportMap = HashMap<Id, Vec<ArrowImportVTableRef>>;
type ExportInferenceMap = HashMap<ExtId, Vec<ArrowExportVTableRef>>;

/// Session-scoped registry of Arrow extension plugins.
///
/// Exporters are stored in two indices: one keyed by Arrow extension Id (used for
/// `execute_arrow` dispatch) and one keyed by Vortex extension Id (used **only** by
/// `to_arrow_field` / `to_arrow_schema` inference, when callers need to translate a Vortex
/// extension `DType` into an Arrow `Field` with no target schema in hand). Importers are
/// keyed by Arrow extension name. The default session pre-registers the builtin UUID
/// plugin; temporal extensions are handled by the canonical Arrow ↔ Vortex path and do not
/// need plugins.
#[derive(Debug)]
pub struct ArrowSession {
    exporters: ArcSwap<ExportMap>,
    exporters_by_vortex: ArcSwap<ExportInferenceMap>,
    importers: ArcSwap<ImportMap>,
}

impl Default for ArrowSession {
    fn default() -> Self {
        let session = Self {
            exporters: ArcSwap::from_pointee(ExportMap::default()),
            exporters_by_vortex: ArcSwap::from_pointee(ExportInferenceMap::default()),
            importers: ArcSwap::from_pointee(ImportMap::default()),
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
        Self::insert(
            &self.exporters,
            exporter.arrow_ext_id(),
            ArrowExportVTableRef::clone(&exporter),
        );
        Self::insert(
            &self.exporters_by_vortex,
            exporter.vortex_ext_id(),
            exporter,
        );
    }

    /// Register an [`ArrowImportVTable`] under its source Arrow extension name.
    pub fn register_importer(&self, importer: ArrowImportVTableRef) {
        Self::insert(&self.importers, importer.arrow_ext_id(), importer);
    }

    fn insert<K, T>(slot: &ArcSwap<HashMap<K, Vec<T>>>, key: K, value: T)
    where
        K: Clone + Eq + std::hash::Hash,
        T: Clone,
    {
        slot.rcu(move |map| {
            let mut next = (**map).clone();
            next.entry(key.clone()).or_default().push(value.clone());
            next
        });
    }

    fn exporters(&self, id: &Id) -> Vec<ArrowExportVTableRef> {
        self.exporters.load().get(id).cloned().unwrap_or_default()
    }

    fn exporters_by_vortex(&self, id: &ExtId) -> Vec<ArrowExportVTableRef> {
        self.exporters_by_vortex
            .load()
            .get(id)
            .cloned()
            .unwrap_or_default()
    }

    fn importers(&self, id: &Id) -> Vec<ArrowImportVTableRef> {
        self.importers.load().get(id).cloned().unwrap_or_default()
    }

    /// Build the Arrow [`Field`] for a Vortex [`DType`].
    ///
    /// For [`DType::Extension`]s, the first plugin registered against the extension's
    /// Vortex Id is consulted to produce a [`Field`] with the appropriate Arrow extension
    /// metadata. All other dtypes go through canonical [`DType::to_arrow_dtype`].
    pub fn to_arrow_field(&self, name: &str, dtype: &DType) -> VortexResult<Field> {
        if let Some(ext) = dtype.as_extension_opt() {
            let exporters = self.exporters_by_vortex(&ext.id());
            if let Some(plugin) = exporters.first() {
                return plugin.to_arrow_field(name, ext);
            }
        }
        Ok(Field::new(
            name,
            dtype.to_arrow_dtype()?,
            dtype.is_nullable(),
        ))
    }

    /// Build the Arrow [`Schema`] for a Vortex top-level [`DType::Struct`], dispatching
    /// extension fields through registered export plugins for inference.
    pub fn to_arrow_schema(&self, dtype: &DType) -> VortexResult<Schema> {
        let DType::Struct(struct_dtype, _) = dtype else {
            vortex_error::vortex_bail!(
                "to_arrow_schema requires a top-level struct dtype, got {dtype}"
            );
        };
        let mut fields = Vec::with_capacity(struct_dtype.names().len());
        for (name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
            fields.push(self.to_arrow_field(name.as_ref(), &field_dtype)?);
        }
        Ok(Schema::new(fields))
    }

    /// Build the Vortex [`DType`] for an Arrow [`Field`].
    ///
    /// Routes through the registered import plugin if the field carries an Arrow extension
    /// name we recognize; otherwise uses the canonical Arrow → Vortex type mapping.
    pub fn from_arrow_field(&self, field: &Field) -> VortexResult<DType> {
        if let Some(name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            let importers = self.importers(&Id::new(name));
            if let Some(plugin) = importers.first() {
                return plugin.from_arrow_field(field);
            }
        }
        // Fall back to handling of canonical types + Vortex builtin canonical types.
        Ok(DType::from_arrow(field))
    }

    /// Build the Vortex [`DType`] for an Arrow [`Schema`], dispatching extension fields
    /// through registered import plugins. The result is a top-level non-nullable struct
    /// matching the schema's fields.
    pub fn from_arrow_schema(&self, schema: &Schema) -> VortexResult<DType> {
        let entries = schema
            .fields()
            .iter()
            .map(|f| {
                self.from_arrow_field(f)
                    .map(|dt| (FieldName::from(f.name().as_str()), dt))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(DType::Struct(
            StructFields::from_iter(entries),
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
    /// If `target` carries an `ARROW:extension:name`, the matching export plugin runs. If no
    /// plugin matches (or all return [`ArrowExport::Unsupported`]), falls back to the
    /// canonical Vortex → Arrow path. With `target = None` the canonical path picks the
    /// array's preferred Arrow type.
    pub fn execute_arrow(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let Some(target_field) = target else {
            return canonical_execute_arrow(array, None, ctx);
        };
        let Some(arrow_ext_name) = target_field.metadata().get(EXTENSION_TYPE_NAME_KEY) else {
            return canonical_execute_arrow(array, Some(target_field.data_type()), ctx);
        };

        let exporters = self.exporters(&Id::new(arrow_ext_name));
        if exporters.is_empty() {
            return canonical_execute_arrow(array, Some(target_field.data_type()), ctx);
        }

        let len = array.len();
        let mut current = array;
        for plugin in &exporters {
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

        // Fallback to canonical execution path
        canonical_execute_arrow(current, Some(target_field.data_type()), ctx)
    }

    /// Decode an Arrow array into a Vortex array.
    ///
    /// Routes through the registered import plugin if `field` carries an Arrow extension
    /// name we recognize, probing each plugin in registration order until one handles the
    /// input or all return [`ArrowImport::Unsupported`]; otherwise uses the canonical
    /// Arrow → Vortex array conversion.
    pub fn from_arrow_array(&self, array: ArrowArrayRef, field: &Field) -> VortexResult<ArrayRef> {
        let Some(extension_name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) else {
            return ArrayRef::from_arrow(array.as_ref(), field.is_nullable());
        };

        let importers = self.importers(&Id::new(extension_name));
        if importers.is_empty() {
            return ArrayRef::from_arrow(array.as_ref(), field.is_nullable());
        }

        let dtype = self.from_arrow_field(field)?;
        let DType::Extension(ext_dtype) = dtype else {
            return ArrayRef::from_arrow(array.as_ref(), field.is_nullable());
        };

        let mut current = array;
        for plugin in &importers {
            match plugin.from_arrow_array(current, &ext_dtype)? {
                ArrowImport::Imported(arr) => return Ok(arr),
                ArrowImport::Unsupported(arr) => current = arr,
            }
        }

        ArrayRef::from_arrow(current.as_ref(), field.is_nullable())
    }
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
    fn arrow(&self) -> Ref<'_, ArrowSession>;
}

impl<S: SessionExt> ArrowSessionExt for S {
    fn arrow(&self) -> Ref<'_, ArrowSession> {
        self.get::<ArrowSession>()
    }
}
