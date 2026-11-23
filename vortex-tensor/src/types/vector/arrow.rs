// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow plugin impls for the [`Vector`] extension type.
//!
//! [`Vector`] stores `FixedSizeList<float, N>` and is round-tripped through Arrow as a
//! `FixedSizeList<float, N>` carrying the `vortex.tensor.vector` extension name on the field
//! metadata. The element layout is identical on both sides, so the conversion is structural.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
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
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::types::vector::Vector;
use crate::types::vector::VectorMetadata;

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

        // The vector dtype requires non-nullable elements even when Arrow declares them nullable.
        let element = session.from_arrow_datatype(elem.data_type(), Nullability::NonNullable)?;
        let metadata = VectorMetadata::try_new(PType::try_from(&element)?, *list_size as u32)?;
        Ok(Some(ExtDType::try_with_vtable(
            Vector,
            metadata,
            field.is_nullable().into(),
        )?))
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowSessionExt;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::tests::SESSION;
    use crate::types::vector::Vector;

    const DIM: u32 = 3;

    fn vector_dtype(nullable: bool) -> DType {
        let metadata = VectorMetadata::try_new(PType::F32, DIM).expect("float elements");
        DType::Extension(
            ExtDType::try_with_vtable(Vector, metadata, nullable.into())
                .expect("vector ext dtype")
                .erased(),
        )
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
    fn from_arrow_array_rejects_non_fsl() {
        let dtype = vector_dtype(false);
        let field = Field::new("embedding", DataType::Int32, false);

        let int_array: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let result = <Vector as ArrowExtVTable>::from_arrow_array(
            int_array,
            &field,
            &dtype.as_extension().clone().downcast::<Vector>(),
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

    /// A field named `vortex.tensor.vector` over `data_type`.
    fn vector_field(data_type: DataType, nullable: bool) -> Field {
        let mut field = Field::new("embedding", data_type, nullable);
        set_arrow_extension_name(&mut field, ARROW_VECTOR_EXTENSION_NAME);
        field
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
        let err = session_with_vector()
            .arrow()
            .from_arrow_field(&vector_field(data_type, false))
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
        let field = vector_field(array.data_type().clone(), true);
        (Arc::new(array), field)
    }

    #[test]
    fn test_import_null_elements_under_null_vector() -> VortexResult<()> {
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

    #[test]
    fn test_import_null_element_in_valid_vector() -> VortexResult<()> {
        let (arrow, field) = arrow_vectors(vec![Some(vec![Some(1.0), None, Some(3.0)])]);
        let imported = session_with_vector()
            .arrow()
            .from_arrow_array(arrow, &field)?;

        let storage = FixedSizeListArray::try_new(
            buffer![1.0f32, 0.0, 3.0].into_array(),
            DIM,
            Validity::AllValid,
            1,
        )?;
        let expected = Vector::try_new_vector_array(storage.into_array())?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// Build a non-nullable vector field with a null vector for use under a null parent row.
    fn non_nullable_vectors_with_nulls() -> (ArrowArrayRef, Field) {
        let (vectors, field) = arrow_vectors(vec![
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]), // A vector under a valid row.
            None,                                        // A null vector under a null row.
        ]);
        (vectors, field.with_nullable(false))
    }

    fn expected_non_nullable_vectors() -> VortexResult<ArrayRef> {
        let storage = FixedSizeListArray::try_new(
            buffer![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0].into_array(),
            DIM,
            Validity::NonNullable,
            2,
        )?;
        Vector::try_new_vector_array(storage.into_array())
    }

    /// A null struct row can contain a null vector in a non-nullable field.
    #[test]
    fn test_import_non_nullable_vector_field_under_null_struct_row() -> VortexResult<()> {
        let (vectors, field) = non_nullable_vectors_with_nulls();
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
            vec![expected_non_nullable_vectors()?],
            2,
            Validity::from_iter([true, false]),
        )?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A null list can contain null vectors in a non-nullable element field.
    #[test]
    fn test_import_non_nullable_vector_elements_under_null_list() -> VortexResult<()> {
        let (vectors, field) = non_nullable_vectors_with_nulls();
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
            expected_non_nullable_vectors()?,
            1,
            Validity::from_iter([true, false]),
            2,
        )?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A valid vector inside a null list can still contain null elements.
    #[test]
    fn test_import_valid_vector_under_null_variable_size_list() -> VortexResult<()> {
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
}
