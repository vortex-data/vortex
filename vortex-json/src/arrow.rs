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
use vortex_array::arrow::ArrowExport;
use vortex_array::arrow::ArrowExportVTable;
use vortex_array::arrow::ArrowImport;
use vortex_array::arrow::ArrowImportVTable;
use vortex_array::arrow::ArrowSession;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
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

        ArrowJson::try_new_from_field_metadata(target.data_type(), target.metadata())?;

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();
        let storage_field = Field::new(
            String::new(),
            target.data_type().clone(),
            target.is_nullable(),
        );
        let session = ctx.session().clone();

        let storage = session
            .arrow()
            .execute_arrow(storage, Some(&storage_field), ctx)?;

        Ok(ArrowExport::Exported(storage))
    }
}

impl ArrowImportVTable for Json {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_JSON
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        if !has_valid_json_extension(field) {
            return Ok(None);
        }

        Ok(Some(DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, DType::Utf8(field.is_nullable().into()))?
                .erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
    ) -> VortexResult<ArrowImport> {
        let DType::Extension(ext_dtype) = dtype else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !ext_dtype.is::<Json>() || !has_valid_json_extension(field) {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = ArrayRef::from_arrow(array.as_ref(), field.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::new(ext_dtype.clone(), storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

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
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Json;
    use crate::initialize;

    /// Export a JSON extension array to Arrow's canonical JSON extension.
    #[test]
    fn exports_json_extension_array_as_arrow_json() -> VortexResult<()> {
        let session = VortexSession::empty();
        initialize(&session);

        let storage = VarBinArray::from_iter(
            [Some("{\"id\":1}"), Some("{\"id\":2}")],
            vortex_array::dtype::DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let ext_dtype = ExtDType::<Json>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();

        dbg!(&ext_dtype);
        let array = ExtensionArray::new(ext_dtype, storage).into_array();

        let field = session.arrow().to_arrow_field("data", array.dtype())?;
        assert_eq!(field.extension_type_name(), Some(ArrowJson::NAME));
        ArrowJson::try_new_from_field_metadata(field.data_type(), field.metadata())?;

        dbg!(&field);

        let exported = session.arrow().execute_arrow(
            array,
            Some(&field),
            &mut session.create_execution_ctx(),
        )?;

        assert!(exported.data_type().is_string());

        dbg!(exported.data_type());

        let strings = exported.as_string_view();
        assert_eq!(strings.value(0), "{\"id\":1}");
        assert_eq!(strings.value(1), "{\"id\":2}");
        Ok(())
    }

    /// Import Arrow's canonical JSON extension as a Vortex JSON extension array.
    #[test]
    fn imports_arrow_json_extension_array_as_vortex_json() -> VortexResult<()> {
        let session = VortexSession::empty();
        initialize(&session);

        let mut field = Field::new("data", DataType::Utf8, false);
        field.try_with_extension_type(ArrowJson::default())?;
        let array = Arc::new(StringArray::from(vec!["{\"id\":1}", "{\"id\":2}"])) as ArrowArrayRef;

        let imported = session.arrow().from_arrow_array(array, &field)?;
        let ext_dtype = imported
            .dtype()
            .as_extension_opt()
            .vortex_expect("expected JSON extension dtype");
        assert!(ext_dtype.is::<Json>());

        let exported = session.arrow().execute_arrow(
            imported,
            Some(&field),
            &mut session.create_execution_ctx(),
        )?;
        let strings = exported.as_string::<i32>();
        assert_eq!(strings.value(0), "{\"id\":1}");
        assert_eq!(strings.value(1), "{\"id\":2}");
        Ok(())
    }
}
