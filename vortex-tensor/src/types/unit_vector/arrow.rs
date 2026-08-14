// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow conversion for [`UnitVector`].
//!
//! Arrow data carrying the `vortex.tensor.unit_vector` extension name is trusted to satisfy the
//! unit-norm refinement. This keeps import structural and zero-copy; callers handling untrusted
//! values must use [`UnitVector::try_new_unit_vector_array`] instead.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::types::unit_vector::UnitVector;

/// Arrow extension name used to identify [`UnitVector`] fields on the wire.
pub const ARROW_UNIT_VECTOR_EXTENSION_NAME: &str = "vortex.tensor.unit_vector";

static ARROW_UNIT_VECTOR: CachedId = CachedId::new(ARROW_UNIT_VECTOR_EXTENSION_NAME);

#[expect(
    clippy::disallowed_types,
    reason = "Arrow's Field::set_metadata requires std::collections::HashMap"
)]
fn unit_vector_extension_metadata() -> std::collections::HashMap<String, String> {
    [(
        EXTENSION_TYPE_NAME_KEY.to_string(),
        ARROW_UNIT_VECTOR_EXTENSION_NAME.to_string(),
    )]
    .into()
}

fn is_supported_float(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    )
}

impl ArrowExportVTable for UnitVector {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_UNIT_VECTOR
    }

    fn vortex_id(&self) -> Id {
        UnitVector.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let DType::Extension(dtype) = dtype else {
            return Ok(None);
        };
        if !dtype.is::<UnitVector>() {
            return Ok(None);
        }

        let mut field = session.to_arrow_field(name, dtype.storage_dtype())?;
        field.set_metadata(unit_vector_extension_metadata());
        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        if !array
            .dtype()
            .as_extension_opt()
            .is_some_and(|ext| ext.is::<UnitVector>())
        {
            return Ok(ArrowExport::Unsupported(array));
        }

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();
        let session = ctx.session().clone();
        let arrow_storage = session.arrow().execute_arrow(storage, Some(target), ctx)?;

        Ok(ArrowExport::Exported(arrow_storage))
    }
}

impl ArrowImportVTable for UnitVector {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_UNIT_VECTOR
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        if field.extension_type_name() != Some(ARROW_UNIT_VECTOR_EXTENSION_NAME) {
            return Ok(None);
        }
        let DataType::FixedSizeList(element, list_size) = field.data_type() else {
            return Ok(None);
        };
        if !is_supported_float(element.data_type()) || element.is_nullable() {
            return Ok(None);
        }

        let storage_dtype = DType::FixedSizeList(
            Arc::new(session.from_arrow_field(element.as_ref())?),
            *list_size as u32,
            field.is_nullable().into(),
        );
        let dtype = ExtDType::try_with_vtable(UnitVector, EmptyMetadata, storage_dtype)?;

        Ok(Some(DType::Extension(dtype.erased())))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        _field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let DType::Extension(dtype) = dtype else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !dtype.is::<UnitVector>() {
            return Ok(ArrowImport::Unsupported(array));
        }
        let DataType::FixedSizeList(element, _) = array.data_type() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !is_supported_float(element.data_type()) {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = session.from_arrow_array(array, dtype.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(dtype.clone(), storage)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;

    use super::*;

    const DIMENSIONS: u32 = 2;

    fn unit_vector_dtype() -> VortexResult<DType> {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            DIMENSIONS,
            Nullability::NonNullable,
        );
        let dtype = ExtDType::<UnitVector>::try_new(EmptyMetadata, storage)?;

        Ok(DType::Extension(dtype.erased()))
    }

    fn session_with_unit_vector() -> ArrowSession {
        let session = ArrowSession::default();
        session.register_exporter(Arc::new(UnitVector));
        session.register_importer(Arc::new(UnitVector));
        session
    }

    #[test]
    fn field_round_trip_preserves_unit_vector() -> VortexResult<()> {
        let session = session_with_unit_vector();
        let dtype = unit_vector_dtype()?;
        let field = session.to_arrow_field("embedding", &dtype)?;

        assert_eq!(
            field.extension_type_name(),
            Some(ARROW_UNIT_VECTOR_EXTENSION_NAME),
        );
        assert_eq!(session.from_arrow_field(&field)?, dtype);
        Ok(())
    }

    #[test]
    fn tagged_import_trusts_the_refinement() -> VortexResult<()> {
        let session = session_with_unit_vector();
        let field = session.to_arrow_field("embedding", &unit_vector_dtype()?)?;
        let values = Arc::new(Float32Array::from(vec![3.0, 4.0]));
        let element = Arc::new(Field::new("item", DataType::Float32, false));
        let arrow: ArrowArrayRef = Arc::new(ArrowFixedSizeListArray::new(
            element,
            DIMENSIONS as i32,
            values,
            None,
        ));

        let imported = session.from_arrow_array(arrow, &field)?;
        assert!(imported.dtype().as_extension().is::<UnitVector>());
        Ok(())
    }
}
