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

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::make_array;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use arrow_schema::extension::ExtensionType;
use tracing::debug;
use tracing::trace;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arc_swap_map::ArcSwapMap;
use crate::arrays::StructArray;
use crate::arrow::FromArrowArray;
use crate::arrow::convert::nulls;
use crate::arrow::convert::remove_nulls;
use crate::arrow::executor::execute_arrow_naive;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::dtype::arrow::FromArrowType;
use crate::dtype::arrow::to_data_type_naive;
use crate::dtype::extension::ExtId;
use crate::extension::datetime::AnyTemporal;
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
    #[allow(clippy::wrong_self_convention)]
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>>;

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
    exporters: ArcSwapMap<Id, Arc<[ArrowExportVTableRef]>>,
    exporters_by_vortex: ArcSwapMap<ExtId, Arc<[ArrowExportVTableRef]>>,
    importers: ArcSwapMap<Id, Arc<[ArrowImportVTableRef]>>,
}

impl Default for ArrowSession {
    fn default() -> Self {
        let session = Self {
            exporters: ArcSwapMap::default(),
            exporters_by_vortex: ArcSwapMap::default(),
            importers: ArcSwapMap::default(),
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
    /// Plugins registered against the field's Arrow extension name are tried in
    /// registration order; the first plugin to return `Some(dtype)` wins. If none
    /// match (or all return `None`), recurses into container types ([`DataType::List`]
    /// family, [`DataType::FixedSizeList`], [`DataType::Struct`]) so extension metadata
    /// on nested element/struct fields is preserved. Leaf types use the canonical
    /// Arrow → Vortex mapping via [`DType::from_arrow`].
    pub fn from_arrow_field(&self, field: &Field) -> VortexResult<DType> {
        if let Some(name) = field.metadata().get(EXTENSION_TYPE_NAME_KEY) {
            for plugin in self.importers(&Id::new(name)).iter() {
                if let Some(dtype) = plugin.from_arrow_field(field)? {
                    return Ok(dtype);
                }
            }
        }
        let nullability: Nullability = field.is_nullable().into();
        Ok(match field.data_type() {
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
            DataType::Struct(fields) => {
                let entries = fields
                    .iter()
                    .map(|f| {
                        self.from_arrow_field(f)
                            .map(|dt| (FieldName::from(f.name().as_str()), dt))
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                DType::Struct(StructFields::from_iter(entries), nullability)
            }
            _ => DType::from_arrow(field),
        })
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
    /// If `target` carries an `ARROW:extension:name`, the plugin registry is probed for one that
    /// can support executing to the target extension type.
    ///
    /// With `target = None` the fallback path picks the array's preferred Arrow physical type
    /// and executes directly into that, ignoring extension types.
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
                            make_array(remove_nulls(col.to_data()))
                        } else {
                            ArrowArrayRef::clone(col)
                        };
                        self.from_arrow_array(inner, child_field.as_ref())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                let validity = nulls(arrow_struct.nulls(), field.is_nullable());
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
                let validity = nulls(list.nulls(), field.is_nullable());
                Ok(crate::arrays::ListArray::try_new(elements, offsets, validity)?.into_array())
            }
            DataType::LargeList(elem_field) => {
                let list = array.as_list::<i64>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable());
                Ok(crate::arrays::ListArray::try_new(elements, offsets, validity)?.into_array())
            }
            DataType::FixedSizeList(elem_field, list_size) => {
                let fsl = array.as_fixed_size_list();
                let elements =
                    self.from_arrow_array(ArrowArrayRef::clone(fsl.values()), elem_field.as_ref())?;
                let validity = nulls(fsl.nulls(), field.is_nullable());
                Ok(crate::arrays::FixedSizeListArray::try_new(
                    elements,
                    *list_size as u32,
                    validity,
                    fsl.len(),
                )?
                .into_array())
            }
            DataType::ListView(elem_field) => {
                let list = array.as_list_view::<i32>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let sizes = list.sizes().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable());
                Ok(
                    crate::arrays::ListViewArray::try_new(elements, offsets, sizes, validity)?
                        .into_array(),
                )
            }
            DataType::LargeListView(elem_field) => {
                let list = array.as_list_view::<i64>();
                let elements = self
                    .from_arrow_array(ArrowArrayRef::clone(list.values()), elem_field.as_ref())?;
                let offsets = list.offsets().clone().into_array();
                let sizes = list.sizes().clone().into_array();
                let validity = nulls(list.nulls(), field.is_nullable());
                Ok(
                    crate::arrays::ListViewArray::try_new(elements, offsets, sizes, validity)?
                        .into_array(),
                )
            }
            _ => ArrayRef::from_arrow(array.as_ref(), field.is_nullable()),
        }
    }
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
    fn arrow(&self) -> Ref<'_, ArrowSession>;
}

impl<S: SessionExt> ArrowSessionExt for S {
    fn arrow(&self) -> Ref<'_, ArrowSession> {
        self.get::<ArrowSession>()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::FixedSizeBinaryArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::Uuid as ArrowUuid;
    use vortex_error::VortexResult;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::uuid::Uuid;
    use crate::extension::uuid::UuidMetadata;

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
    fn execute_arrow_target_none_preserves_top_level_uuid_metadata() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let session = LEGACY_SESSION.arrow();

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
}
