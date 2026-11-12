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
use vortex_arrow::ext_metadata::set_arrow_extension_name;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::types::vector::Vector;

/// Arrow extension name used to identify [`Vector`] fields on the wire.
pub const ARROW_VECTOR_EXTENSION_NAME: &str = "vortex.tensor.vector";

static ARROW_VECTOR: CachedId = CachedId::new(ARROW_VECTOR_EXTENSION_NAME);

fn is_supported_float(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    )
}

impl ArrowExtVTable for Vector {
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::ArrowNative
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_VECTOR
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let mut field = session.to_arrow_field(name, ext_dtype.storage_dtype())?;
        set_arrow_extension_name(&mut field, ARROW_VECTOR_EXTENSION_NAME);

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();

        let arrow_storage = storage.execute_arrow_with_options(Some(target), options, ctx)?;

        Ok(ArrowExport::Exported(arrow_storage))
    }

    fn from_arrow_field(
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
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
            Arc::new(session.from_arrow_field(elem.as_ref())?),
            *list_size as u32,
            field.is_nullable().into(),
        );

        Ok(Some(ExtDType::try_with_vtable(
            Vector,
            EmptyMetadata,
            storage_dtype,
        )?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        _field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let DataType::FixedSizeList(elem, _) = array.data_type() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !is_supported_float(elem.data_type()) {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = session.from_arrow_array(array, ext_dtype.storage_dtype().is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array(),
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowImport;
    use vortex_arrow::ArrowSessionExt;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

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

    /// A session whose only Arrow plugins are the [`Vector`] exporter and importer.
    fn session_with_vector() -> VortexSession {
        let session = VortexSession::empty();
        session.arrow().register_extension(Vector);
        session
    }

    #[test]
    fn to_arrow_field_attaches_extension_metadata() -> VortexResult<()> {
        let vortex_session = session_with_vector();
        let session = vortex_session.arrow();
        let field = session.to_arrow_field("embedding", &vector_dtype(false))?;
        assert_eq!(
            field.extension_type_name(),
            Some(ARROW_VECTOR_EXTENSION_NAME),
        );
        let DataType::FixedSizeList(elem, size) = field.data_type() else {
            panic!("expected FixedSizeList, got {:?}", field.data_type());
        };
        assert_eq!(field.name(), "embedding");
        assert!(!field.is_nullable());
        assert_eq!(*size, DIM as i32);
        assert_eq!(elem.data_type(), &DataType::Float32);
        assert!(!elem.is_nullable());
        Ok(())
    }

    #[test]
    fn from_arrow_field_recovers_vector_dtype() -> VortexResult<()> {
        let vortex_session = session_with_vector();
        let session = vortex_session.arrow();
        let mut arrow_field = Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
            ),
            true,
        );
        set_arrow_extension_name(&mut arrow_field, ARROW_VECTOR_EXTENSION_NAME);
        let dtype = session.from_arrow_field(&arrow_field)?;
        assert_eq!(dtype, vector_dtype(true));
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_top_level_vector() -> VortexResult<()> {
        let vortex_session = session_with_vector();
        let session = vortex_session.arrow();
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
        let vortex_session = session_with_vector();
        let session = vortex_session.arrow();
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
        let vortex_session = session_with_vector();
        let session = vortex_session.arrow();
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
        let arrow = original.clone().execute_arrow(Some(&field), &mut ctx)?;

        assert!(matches!(arrow.data_type(), DataType::FixedSizeList(_, n) if *n == DIM as i32));

        assert_eq!(
            arrow.as_ref(),
            arrow_fsl_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DIM as i32).as_ref()
        );
        let imported = session.from_arrow_array(arrow, &field)?;
        assert_eq!(imported.dtype(), original.dtype());
        vortex_array::assert_arrays_eq!(imported, original, &mut ctx);
        Ok(())
    }

    #[test]
    fn unregistered_session_falls_back_to_canonical_import() -> VortexResult<()> {
        // Session with no Vector plugin must not error on a vector.tensor.vector-tagged field;
        // it should fall through to the canonical Arrow → Vortex path and drop the extension
        // identity, producing the raw FSL storage instead.
        let unregistered = VortexSession::empty();
        let session = unregistered.arrow();
        let mut arrow_field = Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
            ),
            false,
        );
        set_arrow_extension_name(&mut arrow_field, ARROW_VECTOR_EXTENSION_NAME);
        let arrow = arrow_fsl_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DIM as i32);
        let imported = session.from_arrow_array(arrow, &arrow_field)?;
        assert!(
            matches!(imported.dtype(), DType::FixedSizeList(elem, n, _) if **elem == DType::Primitive(PType::F32, Nullability::NonNullable) && *n == DIM),
            "expected raw FSL dtype, got {}",
            imported.dtype()
        );
        let expected = FixedSizeListArray::try_new(
            buffer![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_array(),
            DIM,
            Validity::NonNullable,
            2,
        )?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn execute_arrow_rejects_incompatible_non_vector() -> VortexResult<()> {
        let session = session_with_vector();
        let mut ctx = session.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter([1_i32, 2, 3]).into_array();
        let target = session
            .arrow()
            .to_arrow_field("embedding", &vector_dtype(false))?;

        assert!(primitive.execute_arrow(Some(&target), &mut ctx).is_err());

        Ok(())
    }

    #[test]
    fn from_arrow_array_returns_unsupported_for_non_fsl() -> VortexResult<()> {
        let dtype = vector_dtype(false);
        let field = Field::new("embedding", DataType::Int32, false);

        let int_array: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let result = <Vector as ArrowExtVTable>::from_arrow_array(
            int_array,
            &field,
            &dtype.as_extension().clone().downcast::<Vector>(),
            &VortexSession::empty().arrow(),
        )?;
        assert!(matches!(result, ArrowImport::Unsupported(_)));
        Ok(())
    }

    #[test]
    fn execute_arrow_through_session_with_no_target() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let session = SESSION.arrow();
        let original = sample_vector_array();
        let arrow = original.clone().execute_arrow(None, &mut ctx)?;

        let field = session.to_arrow_field("v", original.dtype())?;
        assert_eq!(
            arrow.as_ref(),
            arrow_fsl_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DIM as i32).as_ref()
        );
        let imported = session.from_arrow_array(arrow, &field)?;
        assert_eq!(imported.dtype(), original.dtype());
        vortex_array::assert_arrays_eq!(imported, original, &mut ctx);
        Ok(())
    }
}
