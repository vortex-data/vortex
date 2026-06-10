// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow import and export support for the JSON extension dtype.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
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
use vortex_array::dtype::arrow::FromArrowType;
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
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let DType::Extension(ext_dtype) = dtype else {
            return Ok(None);
        };
        if !ext_dtype.is::<Json>() {
            return Ok(None);
        }

        let mut field = Field::new(name, DataType::Utf8, dtype.is_nullable());
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
            storage.dtype().is_nullable(),
        );
        let session = ctx.session().clone();
        Ok(ArrowExport::Exported(session.arrow().execute_arrow(
            storage,
            Some(&storage_field),
            ctx,
        )?))
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

        let storage_dtype = DType::from_arrow(field);
        Ok(Some(DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, storage_dtype)?.erased(),
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
