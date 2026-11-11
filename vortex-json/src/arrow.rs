// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow import and export support for the JSON extension dtype.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use arrow_schema::extension::Json as ArrowJson;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExtVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowMetadataPolicy;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::Json;

/// Arrow's canonical JSON extension name cached as a registry id.
static ARROW_JSON: CachedId = CachedId::new(ArrowJson::NAME);

/// Returns whether an Arrow field contains valid canonical JSON extension metadata.
fn has_valid_json_extension(field: &Field) -> bool {
    field.extension_type_name() == Some(ArrowJson::NAME)
        && ArrowJson::try_new_from_field_metadata(field.data_type(), field.metadata()).is_ok()
}

impl ArrowExtVTable for Json {
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::ArrowNative
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_JSON
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
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
        ArrowJson::try_new_from_field_metadata(target.data_type(), target.metadata())?;

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

    fn from_arrow_field(
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
        if !has_valid_json_extension(field) {
            return Ok(None);
        }

        Ok(Some(ExtDType::<Json>::try_new(
            EmptyMetadata,
            DType::Utf8(field.is_nullable().into()),
        )?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        if !has_valid_json_extension(field) {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = session.from_arrow_array(array, field.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::new(ext_dtype.clone().erased(), storage).into_array(),
        ))
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
    use arrow_schema::extension::ExtensionType;
    use arrow_schema::extension::Json as ArrowJson;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
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
            vortex_array::dtype::DType::Utf8(Nullability::NonNullable),
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
            &vortex_array::dtype::DType::Utf8(Nullability::NonNullable)
        );

        let exported = imported.execute_arrow(Some(&field), &mut SESSION.create_execution_ctx())?;
        let strings = exported.as_string::<i32>();
        assert_eq!(
            strings.iter().collect::<Vec<_>>(),
            [Some("{\"id\":1}"), Some("{\"id\":2}")]
        );
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
