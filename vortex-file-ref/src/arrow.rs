// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow import and export for the file reference dtype.
//!
//! A file reference maps to the canonical `arrow.parquet.file` extension type: a struct of
//! nullable fields that a column defines only as it needs them, in any order. Each string field
//! may be `Utf8`, `LargeUtf8` or `Utf8View`, `inline` may be `Binary`, `LargeBinary` or
//! `BinaryView`, and `offset` and `size` are `Int64`. The extension metadata is empty.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::extension::ExtDType;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExtVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowMetadataPolicy;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_arrow::ext_metadata::set_arrow_extension_name;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::FileRef;
use crate::FileRefField;
use crate::FileRefMetadata;

/// The name of the canonical Arrow extension type for file references.
pub const ARROW_FILE_EXTENSION_NAME: &str = "arrow.parquet.file";

/// [`ARROW_FILE_EXTENSION_NAME`] cached as a registry id.
static ARROW_FILE: CachedId = CachedId::new(ARROW_FILE_EXTENSION_NAME);

/// Whether `data_type` is an Arrow storage type that the spec allows for `field`.
fn is_allowed_arrow_type(field: FileRefField, data_type: &DataType) -> bool {
    match field {
        FileRefField::Uri | FileRefField::ContentType | FileRefField::Checksum => matches!(
            data_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ),
        FileRefField::Offset | FileRefField::Size => matches!(data_type, DataType::Int64),
        FileRefField::Inline => matches!(
            data_type,
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        ),
    }
}

/// Checks a field that names `arrow.parquet.file` against the canonical spec and returns the
/// fields that it defines.
///
/// The import and the export both apply this check, so a file reference column exports back to
/// the field that it imports from. The spec requires a struct of nullable, uniquely named fields
/// from a fixed set, with at least one field, and empty extension metadata. A missing metadata key
/// reads as empty metadata, as Arrow C++ reads it.
fn validate_arrow_file_field(field: &Field) -> VortexResult<FileRefMetadata> {
    if let Some(metadata) = field.extension_type_metadata() {
        vortex_ensure!(
            metadata.is_empty(),
            "{ARROW_FILE_EXTENSION_NAME} metadata must be empty, got {metadata:?}"
        );
    }
    let DataType::Struct(children) = field.data_type() else {
        vortex_bail!(
            "{ARROW_FILE_EXTENSION_NAME} storage must be a struct, got {}",
            field.data_type()
        );
    };

    let fields = children
        .iter()
        .map(|child| {
            let name = child.name();
            let file_field = FileRefField::from_name(name).ok_or_else(|| {
                vortex_err!("{ARROW_FILE_EXTENSION_NAME} has unknown field {name:?}")
            })?;
            vortex_ensure!(
                is_allowed_arrow_type(file_field, child.data_type()),
                "{ARROW_FILE_EXTENSION_NAME} field {name} cannot have type {}",
                child.data_type()
            );
            vortex_ensure!(
                child.is_nullable(),
                "{ARROW_FILE_EXTENSION_NAME} field {name} must be nullable"
            );
            if let Some(extension) = child.extension_type_name() {
                vortex_bail!(
                    "{ARROW_FILE_EXTENSION_NAME} field {name} must not be an extension type, got {extension}"
                );
            }
            Ok(file_field)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    FileRefMetadata::try_new(fields)
}

impl ArrowExtVTable for FileRef {
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::ArrowNative
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_FILE
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let mut field = session.to_arrow_field(name, ext_dtype.storage_dtype())?;
        set_arrow_extension_name(&mut field, ARROW_FILE_EXTENSION_NAME);
        field
            .metadata_mut()
            .insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), String::new());
        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let target_metadata = validate_arrow_file_field(target)?;
        let ext_dtype = array.dtype().as_extension().metadata::<FileRef>();
        vortex_ensure!(
            target_metadata == *ext_dtype,
            "cannot export file references with fields [{ext_dtype}] to {ARROW_FILE_EXTENSION_NAME} fields [{target_metadata}]"
        );

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
        let metadata = validate_arrow_file_field(field)?;
        Ok(Some(ExtDType::try_new(metadata, field.is_nullable().into())?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        _field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        session
            .from_arrow_storage(array, &ext_dtype.clone().erased())
            .map(ArrowImport::Imported)
    }
}
