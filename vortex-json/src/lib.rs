// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! Extension type and related functionality for a JSON extension type for Vortex.

mod arrow;
mod dtype;

use std::sync::Arc;

pub use dtype::Json;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_session::VortexSession;

/// Register JSON extension support with a session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Json);
    session.arrow().register_exporter(Arc::new(Json));
    session.arrow().register_importer(Arc::new(Json));
}

#[cfg(test)]
mod tests {
    //! Tests for JSON extension Arrow export.

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
        let array = ExtensionArray::new(ext_dtype, storage).into_array();

        let field = session.arrow().to_arrow_field("data", array.dtype())?;
        assert_eq!(field.extension_type_name(), Some(ArrowJson::NAME));
        ArrowJson::try_new_from_field_metadata(field.data_type(), field.metadata())?;

        let exported = session.arrow().execute_arrow(
            array,
            Some(&field),
            &mut session.create_execution_ctx(),
        )?;
        assert_eq!(exported.data_type(), &DataType::Utf8);

        let strings = exported.as_string::<i32>();
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
