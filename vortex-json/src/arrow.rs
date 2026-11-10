// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow import and export support for the JSON extension dtype.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use arrow_schema::extension::Json as ArrowJson;
use serde_json::Map;
use serde_json::Value;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::Json;

/// Arrow's canonical JSON extension name cached as a registry id.
static ARROW_JSON: CachedId = CachedId::new(ArrowJson::NAME);

/// Checks a field that names `arrow.json` against the canonical spec.
///
/// The import and the export both apply this check, so a JSON column exports back to the field
/// that it imports from. The spec reserves the fields of the metadata object for later additions
/// that a reader does not need, and Arrow C++ reads a missing key as empty metadata. The arrow-rs
/// check rejects both, so this checks the metadata against the spec itself:
/// <https://arrow.apache.org/docs/format/CanonicalExtensions.html#json>
fn validate_arrow_json_field(field: &Field) -> VortexResult<()> {
    ArrowJson::default().supports_data_type(field.data_type())?;

    if let Some(metadata) = field
        .extension_type_metadata()
        .filter(|metadata| !metadata.is_empty())
    {
        vortex_ensure!(
            serde_json::from_str::<Map<String, Value>>(metadata).is_ok(),
            "arrow.json metadata must be empty or a JSON object, got {metadata:?}"
        );
    }

    Ok(())
}

impl ArrowExportVTable for Json {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_JSON
    }

    fn vortex_id(&self) -> Id {
        Json.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let DType::Extension(ext_dtype) = dtype else {
            return Ok(None);
        };
        if !ext_dtype.is::<Json>() {
            return Ok(None);
        }

        let mut field = session.to_arrow_field(name, ext_dtype.storage_dtype())?;
        field
            .try_with_extension_type(ArrowJson::default())
            .vortex_expect("Utf8 is a valid storage type for Arrow JSON");
        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_json = array
            .dtype()
            .as_extension_opt()
            .map(|ext_dtype| ext_dtype.is::<Json>())
            .unwrap_or(false);
        if !is_json {
            return Ok(ArrowExport::Unsupported(array));
        }

        validate_arrow_json_field(target)?;

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();
        let storage_field = Field::new(
            String::new(),
            target.data_type().clone(),
            target.is_nullable(),
        );

        let storage = storage.execute_arrow_with_options(Some(&storage_field), options, ctx)?;

        Ok(ArrowExport::Exported(storage))
    }
}

impl ArrowImportVTable for Json {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_JSON
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        validate_arrow_json_field(field)?;

        Ok(Some(DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, DType::Utf8(field.is_nullable().into()))?
                .erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        _field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let Some(ext_dtype) = dtype.as_extension_opt().filter(|ext| ext.is::<Json>()) else {
            return Ok(ArrowImport::Unsupported(array));
        };
        session
            .from_arrow_storage(array, ext_dtype)
            .map(ArrowImport::Imported)
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Array;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StringArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use arrow_schema::extension::ExtensionType;
    use arrow_schema::extension::Json as ArrowJson;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_arrow::ArrowExportOptions;
    use vortex_arrow::ArrowSessionExt;
    use vortex_arrow::CompactBuffers;
    use vortex_arrow::ExecuteArrowExt;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Json;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        initialize(&session);
        session
    });

    /// Export a JSON extension array to Arrow's canonical JSON extension.
    #[test]
    fn exports_json_extension_array_as_arrow_json() -> VortexResult<()> {
        let storage = VarBinArray::from_iter(
            [Some("{\"id\":1}"), Some("{\"id\":2}")],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let ext_dtype = ExtDType::<Json>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();

        let array = ExtensionArray::new(ext_dtype, storage).into_array();

        let field = SESSION.arrow().to_arrow_field("data", array.dtype())?;
        assert_eq!(field.extension_type_name(), Some(ArrowJson::NAME));
        assert_eq!(field.name(), "data");
        assert!(!field.is_nullable());
        ArrowJson::try_new_from_field_metadata(field.data_type(), field.metadata())?;

        let exported = array.execute_arrow(Some(&field), &mut SESSION.create_execution_ctx())?;

        assert!(exported.data_type().is_string());

        let strings = exported.as_string_view();
        assert_eq!(
            strings.iter().collect::<Vec<_>>(),
            [Some("{\"id\":1}"), Some("{\"id\":2}")]
        );
        Ok(())
    }

    /// Import Arrow's canonical JSON extension as a Vortex JSON extension array.
    #[test]
    fn imports_arrow_json_extension_array_as_vortex_json() -> VortexResult<()> {
        let mut field = Field::new("data", DataType::Utf8, false);
        field.try_with_extension_type(ArrowJson::default())?;
        let array = Arc::new(StringArray::from(vec!["{\"id\":1}", "{\"id\":2}"])) as ArrowArrayRef;

        let imported = SESSION.arrow().from_arrow_array(array, &field)?;
        let ext_dtype = imported
            .dtype()
            .as_extension_opt()
            .vortex_expect("expected JSON extension dtype");
        assert!(ext_dtype.is::<Json>());
        assert_eq!(
            ext_dtype.storage_dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let exported = imported.execute_arrow(Some(&field), &mut SESSION.create_execution_ctx())?;
        let strings = exported.as_string::<i32>();
        assert_eq!(
            strings.iter().collect::<Vec<_>>(),
            [Some("{\"id\":1}"), Some("{\"id\":2}")]
        );
        Ok(())
    }

    /// A field that names `arrow.json` over `data_type`, with `metadata` under the metadata key when
    /// it is given.
    fn arrow_json_field(data_type: DataType, metadata: Option<&str>) -> Field {
        let name = (
            EXTENSION_TYPE_NAME_KEY.to_string(),
            ArrowJson::NAME.to_string(),
        );
        let metadata = metadata.map(|metadata| {
            (
                EXTENSION_TYPE_METADATA_KEY.to_string(),
                metadata.to_string(),
            )
        });
        Field::new("data", data_type, false)
            .with_metadata([name].into_iter().chain(metadata).collect())
    }

    /// A field that names `arrow.json` imports as JSON when it follows the canonical spec, which
    /// allows a missing metadata key and reserves the fields of the metadata object for later
    /// additions. Any other field fails to import instead of importing as plain storage.
    #[test]
    fn imports_fields_in_the_arrow_json_spec() -> VortexResult<()> {
        let json = DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, DType::Utf8(Nullability::NonNullable))?
                .erased(),
        );
        let fields = [
            (DataType::Utf8, None, true),                    // No metadata key.
            (DataType::Utf8, Some("{}"), true),              // An empty object.
            (DataType::Utf8, Some(r#"{"later": 1}"#), true), // An object with a later field.
            (DataType::Utf8, Some("garbage"), false),        // Metadata that is not an object.
            (DataType::Int32, Some(""), false),              // Storage that is not a string.
        ];
        for (data_type, metadata, imports) in fields {
            assert_eq!(
                SESSION
                    .arrow()
                    .from_arrow_field(&arrow_json_field(data_type, metadata))
                    .ok(),
                imports.then(|| json.clone()),
                "unexpected import of arrow.json metadata {metadata:?}"
            );
        }
        Ok(())
    }

    /// A JSON column exports back to the field that it imports from, also for metadata that the
    /// arrow-rs check rejects: a missing key and an object with a later field.
    #[test]
    fn exports_to_fields_in_the_arrow_json_spec() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Arc::new(StringArray::from(vec!["{\"id\":1}"])) as ArrowArrayRef;
        for metadata in [None, Some(r#"{"later": 1}"#)] {
            let field = arrow_json_field(DataType::Utf8, metadata);
            let imported = SESSION
                .arrow()
                .from_arrow_array(Arc::clone(&array), &field)?;
            let exported = imported.execute_arrow(Some(&field), &mut ctx)?;
            assert_eq!(
                exported.as_ref(),
                array.as_ref(),
                "unexpected export to arrow.json metadata {metadata:?}"
            );
        }
        Ok(())
    }

    /// The plugin must forward export options to its storage export.
    #[test]
    fn export_options_reach_json_storage() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let views = VarBinViewArray::from_iter_str((0..64).map(|i| format!("{{\"id\":{i:>32}}}")));
        let backing = views.data_buffers()[0].as_host().as_ptr();
        // Rows far apart in one buffer, so compaction has to copy.
        let storage = views
            .into_array()
            .take(PrimitiveArray::from_iter([0u32, 48]).into_array())?;
        let ext_dtype = ExtDType::<Json>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
        let array = ExtensionArray::new(ext_dtype, storage).into_array();
        let field = SESSION.arrow().to_arrow_field("data", array.dtype())?;

        let retains_backing = |options: &ArrowExportOptions,
                               ctx: &mut vortex_array::ExecutionCtx|
         -> VortexResult<bool> {
            let exported = array
                .clone()
                .execute_arrow_with_options(Some(&field), options, ctx)?;
            assert_eq!(
                exported
                    .as_string_view()
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect::<Vec<_>>(),
                [
                    Some(format!("{{\"id\":{:>32}}}", 0)),
                    Some(format!("{{\"id\":{:>32}}}", 48))
                ]
            );
            Ok(exported
                .as_string_view()
                .data_buffers()
                .iter()
                .any(|buffer| buffer.as_ptr() == backing))
        };

        let uncompacted = ArrowExportOptions::default().with(CompactBuffers(false));
        assert!(retains_backing(&uncompacted, &mut ctx)?);
        assert!(!retains_backing(&ArrowExportOptions::default(), &mut ctx)?);
        Ok(())
    }
}
