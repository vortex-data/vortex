// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow plugin impls for the [`Vector`] extension type.
//!
//! [`Vector`] stores `FixedSizeList<float, N>` and is round-tripped through Arrow as a
//! `FixedSizeList<float, N>` carrying the `vortex.tensor.vector` extension name on the field
//! metadata. The element layout is identical on both sides, so the conversion is structural.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
        options: &ArrowExportOptions,
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

        let arrow_storage = storage.execute_arrow_with_options(Some(target), options, ctx)?;

        Ok(ArrowExport::Exported(arrow_storage))
    }
}

impl ArrowImportVTable for Vector {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_VECTOR
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        let DataType::FixedSizeList(elem, list_size) = field.data_type() else {
            vortex_bail!(
                "Arrow vector storage must be a FixedSizeList, got {}",
                field.data_type()
            );
        };
        vortex_ensure!(
            is_supported_float(elem.data_type()),
            "Arrow vector elements must be floats, got {}",
            elem.data_type()
        );

        // The elements are non-nullable even when Arrow marks them nullable, and
        // `ArrowSession::from_arrow_storage` imports the array in that form.
        let element = session.from_arrow_datatype(elem.data_type(), Nullability::NonNullable)?;
        let storage_dtype = DType::FixedSizeList(
            Arc::new(element),
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
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let Some(ext_dtype) = dtype.as_extension_opt().filter(|ext| ext.is::<Vector>()) else {
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

    use arrow_array::Array;
    use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
    use arrow_array::Float32Array;
    use arrow_array::Int32Array;
    use arrow_array::ListArray as ArrowListArray;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::builder::NullBufferBuilder;
    use arrow_array::builder::OffsetBufferBuilder;
    use arrow_array::types::Float32Type;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowExport;
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
        session.arrow().register_exporter(Arc::new(Vector));
        session.arrow().register_importer(Arc::new(Vector));
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
        let arrow_field = Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
            ),
            true,
        )
        .with_metadata(vector_extension_metadata());
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
        arrow_field.set_metadata(vector_extension_metadata());
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
    fn execute_arrow_returns_unsupported_for_non_vector() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter([1_i32, 2, 3]).into_array();
        let target = Field::new("ints", DataType::Int32, false);
        let result = <Vector as ArrowExportVTable>::execute_arrow(
            &Vector,
            primitive.clone(),
            &target,
            &ArrowExportOptions::default(),
            &mut ctx,
        )?;
        assert!(
            matches!(result, ArrowExport::Unsupported(arr) if arr.dtype() == primitive.dtype())
        );
        Ok(())
    }

    #[test]
    fn from_arrow_array_rejects_non_fsl() {
        let dtype = vector_dtype(false);
        let field = Field::new("embedding", DataType::Int32, false);

        let int_array: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let result = <Vector as ArrowImportVTable>::from_arrow_array(
            &Vector,
            int_array,
            &field,
            &dtype,
            &VortexSession::empty().arrow(),
        );
        let Err(err) = result else {
            panic!("an Int32 array must fail the vector import");
        };
        assert!(
            err.to_string()
                .contains("Arrow storage of vortex.tensor.vector must convert to"),
            "unexpected error: {err}"
        );
    }

    /// A field that names `vortex.tensor.vector` but cannot store vectors fails to import instead
    /// of importing as plain storage.
    #[rstest]
    #[case::int32(DataType::Int32)]
    #[case::int_elements(DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Int32, false)),
        DIM as i32,
    ))]
    #[case::list(DataType::List(Arc::new(Field::new("item", DataType::Float32, false))))]
    fn from_arrow_field_rejects_other_storage(#[case] data_type: DataType) {
        let field =
            Field::new("embedding", data_type, false).with_metadata(vector_extension_metadata());
        let err = session_with_vector()
            .arrow()
            .from_arrow_field(&field)
            .expect_err("a field that cannot store vectors must fail to import");
        assert!(
            err.to_string().contains("as vortex.tensor.vector"),
            "unexpected error: {err}"
        );
    }

    /// Arrow vectors with nullable elements, built the way pyarrow builds them: a null vector
    /// holds null elements.
    fn arrow_vectors(vectors: Vec<Option<Vec<Option<f32>>>>) -> (ArrowArrayRef, Field) {
        let array =
            ArrowFixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, DIM as i32);
        let field = Field::new("embedding", array.data_type().clone(), true)
            .with_metadata(vector_extension_metadata());
        (Arc::new(array), field)
    }

    /// Nullable elements import as the non-nullable elements of `Vector`, and the null elements
    /// of a null vector are dropped.
    #[test]
    fn nullable_elements_import_as_vector() -> VortexResult<()> {
        let vortex_session = session_with_vector();
        let session = vortex_session.arrow();
        let (arrow, field) = arrow_vectors(vec![
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]), // A valid vector.
            None,                                        // A null vector of null elements.
            Some(vec![Some(4.0), Some(5.0), Some(6.0)]), // A valid vector.
        ]);
        assert_eq!(session.from_arrow_field(&field)?, vector_dtype(true));

        let imported = session.from_arrow_array(arrow, &field)?;
        let storage = FixedSizeListArray::try_new(
            buffer![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0].into_array(),
            DIM,
            Validity::from_iter([true, false, true]),
            3,
        )?;
        let expected = Vector::try_new_vector_array(storage.into_array())?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A null element of a valid vector is outside the domain of `Vector`, so the import fails.
    #[test]
    fn null_element_of_valid_vector_fails() {
        let (arrow, field) = arrow_vectors(vec![Some(vec![Some(1.0), None, Some(3.0)])]);
        let err = session_with_vector()
            .arrow()
            .from_arrow_array(arrow, &field)
            .expect_err("a null element of a valid vector must fail the import");
        assert!(
            err.to_string()
                .contains("Failed to import Arrow storage as vortex.tensor.vector"),
            "unexpected error: {err}"
        );
    }

    /// Vectors that Arrow marks non-nullable, as Arrow kernels such as `take` with null indices
    /// build them under a null parent row: the vector and its nullable-marked elements are null.
    fn masked_vectors() -> (ArrowArrayRef, Field) {
        let (vectors, field) = arrow_vectors(vec![
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]), // A vector under a valid row.
            None,                                        // A null vector under a null row.
        ]);
        (vectors, field.with_nullable(false))
    }

    /// The import of [`masked_vectors`], where the dropped nulls become placeholders.
    fn imported_masked_vectors() -> VortexResult<ArrayRef> {
        let storage = FixedSizeListArray::try_new(
            buffer![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0].into_array(),
            DIM,
            Validity::NonNullable,
            2,
        )?;
        Vector::try_new_vector_array(storage.into_array())
    }

    /// A vector field that Arrow marks non-nullable holds nulls at every level under a null row
    /// of its struct. No row reads them, so the import drops them.
    #[test]
    fn vector_field_under_null_struct_row_imports() -> VortexResult<()> {
        let (vectors, field) = masked_vectors();
        let rows = ArrowStructArray::try_new(
            vec![field].into(),
            vec![Arc::clone(&vectors)],
            vectors.nulls().cloned(),
        )?;
        let imported = session_with_vector()
            .arrow()
            .from_arrow_array(Arc::new(rows), true)?;

        let expected = StructArray::try_new(
            FieldNames::from(["embedding"]),
            vec![imported_masked_vectors()?],
            2,
            Validity::from_iter([true, false]),
        )?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// Vectors that are the non-nullable elements of a fixed-size list hold nulls at every level
    /// under a null list, and the import drops them.
    #[test]
    fn vector_elements_under_null_list_import() -> VortexResult<()> {
        let (vectors, field) = masked_vectors();
        let lists = ArrowFixedSizeListArray::try_new(
            Arc::new(field),
            1,
            Arc::clone(&vectors),
            vectors.nulls().cloned(),
        )?;
        let imported = session_with_vector()
            .arrow()
            .from_arrow_array(Arc::new(lists), true)?;

        let expected = FixedSizeListArray::try_new(
            imported_masked_vectors()?,
            1,
            Validity::from_iter([true, false]),
            2,
        )?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A valid vector in a null list holds null elements. No row reads them, so the import drops
    /// them.
    #[test]
    fn vector_in_null_list_row_imports() -> VortexResult<()> {
        let (vectors, field) = arrow_vectors(vec![
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]), // A vector in a valid list.
            Some(vec![None, None, None]),                // A vector in a null list.
        ]);
        let mut offsets = OffsetBufferBuilder::new(2);
        offsets.push_length(1);
        offsets.push_length(1);
        let mut list_nulls = NullBufferBuilder::new(2);
        list_nulls.append_non_null();
        list_nulls.append_null();
        let lists = ArrowListArray::try_new(
            Arc::new(field),
            offsets.finish(),
            vectors,
            list_nulls.finish(),
        )?;
        let imported = session_with_vector()
            .arrow()
            .from_arrow_array(Arc::new(lists), true)?;

        let mut ctx = SESSION.create_execution_ctx();
        let storage = FixedSizeListArray::try_new(
            buffer![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0].into_array(),
            DIM,
            Validity::AllValid,
            2,
        )?;
        let expected = ListArray::try_new(
            Vector::try_new_vector_array(storage.into_array())?,
            buffer![0i32, 1, 2].into_array(),
            Validity::from_iter([true, false]),
            &mut ctx,
        )?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn from_arrow_array_returns_unsupported_for_non_vector_dtype() -> VortexResult<()> {
        use vortex_uuid::Uuid;
        use vortex_uuid::UuidMetadata;
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
