// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow plugin impls for the [`Vector`] extension type.
//!
//! [`Vector`] stores `FixedSizeList<float, N>` and is round-tripped through Arrow as a
//! `FixedSizeList<float, N>` carrying the `vortex.tensor.vector` extension name on the field
//! metadata. The element layout is identical on both sides, so the conversion is structural.

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
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::types::vector::Vector;

/// Arrow extension name used to identify [`Vector`] fields on the wire.
pub const ARROW_VECTOR_EXTENSION_NAME: &str = "vortex.tensor.vector";

static ARROW_VECTOR: CachedId = CachedId::new(ARROW_VECTOR_EXTENSION_NAME);

#[expect(
    clippy::disallowed_types,
    reason = "Arrow's Field::set_metadata requires std::collections::HashMap"
)]
fn vector_extension_metadata() -> std::collections::HashMap<String, String> {
    [(
        EXTENSION_TYPE_NAME_KEY.to_string(),
        ARROW_VECTOR_EXTENSION_NAME.to_string(),
    )]
    .into()
}

fn is_supported_float(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    )
}

impl ArrowExportVTable for Vector {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_VECTOR
    }

    fn vortex_id(&self) -> Id {
        Vector.id()
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

        if !dtype.is::<Vector>() {
            return Ok(None);
        }

        // Delegate to Arrow encoding of storage type.
        let mut field = session.to_arrow_field(name, dtype.storage_dtype())?;
        field.set_metadata(vector_extension_metadata());
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
            .is_some_and(|ext| ext.is::<Vector>())
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

impl ArrowImportVTable for Vector {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_VECTOR
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        if field.extension_type_name() != Some(ARROW_VECTOR_EXTENSION_NAME) {
            return Ok(None);
        }
        let DataType::FixedSizeList(elem, list_size) = field.data_type() else {
            return Ok(None);
        };
        if !is_supported_float(elem.data_type()) || elem.is_nullable() {
            return Ok(None);
        }

        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::from_arrow(elem.as_ref())),
            *list_size as u32,
            field.is_nullable().into(),
        );
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(Vector, EmptyMetadata, storage_dtype)?.erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        _field: &Field,
        dtype: &DType,
    ) -> VortexResult<ArrowImport> {
        let DType::Extension(dtype) = dtype else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !dtype.is::<Vector>() {
            return Ok(ArrowImport::Unsupported(array));
        }
        let DataType::FixedSizeList(elem, _) = array.data_type() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !is_supported_float(elem.data_type()) {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = ArrayRef::from_arrow(array.as_ref() as &dyn Array, dtype.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(dtype.clone(), storage)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use arrow_array::Int32Array;
    use arrow_schema::Field;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrow::ArrowExport;
    use vortex_array::arrow::ArrowImport;
    use vortex_array::arrow::ArrowSession;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::tests::SESSION;
    use crate::types::vector::Vector;

    const DIM: u32 = 3;

    fn vector_dtype(nullable: bool) -> DType {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            DIM,
            nullable.into(),
        );
        DType::Extension(
            ExtDType::try_with_vtable(Vector, EmptyMetadata, storage)
                .expect("vector ext dtype")
                .erased(),
        )
    }

    fn sample_vector_array() -> ArrayRef {
        let values = buffer![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_array();
        let fsl = FixedSizeListArray::try_new(values, DIM, Validity::NonNullable, 2).expect("fsl");
        Vector::try_new_vector_array(fsl.into_array()).expect("vector ext")
    }

    fn arrow_fsl_f32(values: Vec<f32>, dim: i32) -> ArrowArrayRef {
        let values = Float32Array::from(values);
        let field = Field::new("item", DataType::Float32, false);
        Arc::new(ArrowFixedSizeListArray::new(
            Arc::new(field),
            dim,
            Arc::new(values),
            None,
        ))
    }

    fn session_with_vector() -> ArrowSession {
        let session = ArrowSession::default();
        session.register_exporter(Arc::new(Vector));
        session.register_importer(Arc::new(Vector));
        session
    }

    #[test]
    fn to_arrow_field_attaches_extension_metadata() -> VortexResult<()> {
        let session = session_with_vector();
        let field = session.to_arrow_field("embedding", &vector_dtype(false))?;
        assert_eq!(
            field.extension_type_name(),
            Some(ARROW_VECTOR_EXTENSION_NAME),
        );
        let DataType::FixedSizeList(elem, size) = field.data_type() else {
            panic!("expected FixedSizeList, got {:?}", field.data_type());
        };
        assert_eq!(*size, DIM as i32);
        assert_eq!(elem.data_type(), &DataType::Float32);
        assert!(!elem.is_nullable());
        Ok(())
    }

    #[test]
    fn from_arrow_field_recovers_vector_dtype() -> VortexResult<()> {
        let session = session_with_vector();
        let arrow_field = session.to_arrow_field("embedding", &vector_dtype(true))?;
        let dtype = session.from_arrow_field(&arrow_field)?;
        assert_eq!(dtype, vector_dtype(true));
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_top_level_vector() -> VortexResult<()> {
        let session = session_with_vector();
        let dtype = DType::Struct(
            StructFields::from_iter([(FieldName::from("embedding"), vector_dtype(false))]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&dtype)?;
        let roundtripped = session.from_arrow_schema(&schema)?;
        assert_eq!(roundtripped, dtype);
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_nested_struct_vector() -> VortexResult<()> {
        let session = session_with_vector();
        let inner = DType::Struct(
            StructFields::from_iter([(FieldName::from("embedding"), vector_dtype(true))]),
            Nullability::NonNullable,
        );
        let outer = DType::Struct(
            StructFields::from_iter([(FieldName::from("payload"), inner)]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&outer)?;
        let roundtripped = session.from_arrow_schema(&schema)?;
        assert_eq!(roundtripped, outer);
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_list_of_vector() -> VortexResult<()> {
        let session = session_with_vector();
        let dtype = DType::Struct(
            StructFields::from_iter([(
                FieldName::from("embeddings"),
                DType::List(Arc::new(vector_dtype(false)), Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&dtype)?;
        let roundtripped = session.from_arrow_schema(&schema)?;
        assert_eq!(roundtripped, dtype);
        Ok(())
    }

    #[test]
    fn array_roundtrip_through_session() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();
        let original = sample_vector_array();
        let field = session.to_arrow_field("embedding", original.dtype())?;
        let arrow = session.execute_arrow(original.clone(), Some(&field), &mut ctx)?;

        assert!(matches!(arrow.data_type(), DataType::FixedSizeList(_, n) if *n == DIM as i32));

        let imported = session.from_arrow_array(arrow, &field)?;
        assert_eq!(imported.dtype(), original.dtype());
        vortex_array::assert_arrays_eq!(imported, original);
        Ok(())
    }

    #[test]
    fn unregistered_session_falls_back_to_canonical_import() -> VortexResult<()> {
        // Session with no Vector plugin must not error on a vector.tensor.vector-tagged field;
        // it should fall through to the canonical Arrow → Vortex path and drop the extension
        // identity, producing the raw FSL storage instead.
        let session = ArrowSession::default();
        let mut arrow_field = Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
            ),
            false,
        );
        arrow_field.set_metadata(vector_extension_metadata());
        let arrow = arrow_fsl_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DIM as i32);
        let imported = session.from_arrow_array(arrow, &arrow_field)?;
        assert!(
            matches!(imported.dtype(), DType::FixedSizeList(elem, n, _) if **elem == DType::Primitive(PType::F32, Nullability::NonNullable) && *n == DIM),
            "expected raw FSL dtype, got {}",
            imported.dtype()
        );
        Ok(())
    }

    #[test]
    fn execute_arrow_returns_unsupported_for_non_vector() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter([1_i32, 2, 3]).into_array();
        let target = Field::new("ints", DataType::Int32, false);
        let result = <Vector as ArrowExportVTable>::execute_arrow(
            &Vector,
            primitive.clone(),
            &target,
            &mut ctx,
        )?;
        assert!(
            matches!(result, ArrowExport::Unsupported(arr) if arr.dtype() == primitive.dtype())
        );
        Ok(())
    }

    #[test]
    fn from_arrow_array_returns_unsupported_for_non_fsl() -> VortexResult<()> {
        let dtype = vector_dtype(false);
        let field = Field::new("embedding", DataType::Int32, false);

        let int_array: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let result =
            <Vector as ArrowImportVTable>::from_arrow_array(&Vector, int_array, &field, &dtype)?;
        assert!(matches!(result, ArrowImport::Unsupported(_)));
        Ok(())
    }

    #[test]
    fn from_arrow_array_returns_unsupported_for_non_vector_dtype() -> VortexResult<()> {
        use vortex_array::extension::uuid::Uuid;
        use vortex_array::extension::uuid::UuidMetadata;
        let uuid_storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            Nullability::NonNullable,
        );
        let uuid_ext =
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), uuid_storage)?.erased();

        let fsl_arrow = arrow_fsl_f32(vec![1.0, 2.0, 3.0], DIM as i32);
        let field = Field::new("embedding", fsl_arrow.data_type().clone(), false);
        let result = <Vector as ArrowImportVTable>::from_arrow_array(
            &Vector,
            fsl_arrow,
            &field,
            &DType::Extension(uuid_ext),
        )?;
        assert!(matches!(result, ArrowImport::Unsupported(_)));
        Ok(())
    }

    #[test]
    fn execute_arrow_through_session_with_no_target() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();
        let original = sample_vector_array();
        let arrow = session.execute_arrow(original.clone(), None, &mut ctx)?;

        let field = session.to_arrow_field("v", original.dtype())?;
        let imported = session.from_arrow_array(arrow, &field)?;
        assert_eq!(imported.dtype(), original.dtype());
        vortex_array::assert_arrays_eq!(imported, original);
        Ok(())
    }
}
