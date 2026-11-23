// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::Claim;
use vortex_array::dtype::Claims;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::types::vector::Vector;
use crate::types::vector::VectorMetadata;

impl ExtVTable for Vector {
    type Metadata = VectorMetadata;
    type StoredMetadata = ();

    // TODO(connor): This is just a placeholder for now.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.tensor.vector");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        Ok(())
    }

    fn metadata_from_storage(
        &self,
        _stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        let DType::FixedSizeList(element_dtype, dimensions, _nullability) = storage_dtype else {
            vortex_bail!("Vector storage dtype must be a FixedSizeList, got {storage_dtype}");
        };
        let DType::Primitive(element_ptype, _) = element_dtype.as_ref() else {
            vortex_bail!("Vector element dtype must be a float, got {element_dtype}");
        };

        VectorMetadata::try_new(*element_ptype, *dimensions)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(
                metadata.element_ptype(),
                Nullability::NonNullable,
            )),
            metadata.dimensions(),
            Nullability::NonNullable,
        )
    }

    fn claims(_metadata: &Self::Metadata) -> Claims {
        Claims::none().with(Claim::Eq)
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_functions::list_contains::ListContains;
    use vortex_session::VortexSession;

    use crate::types::vector::Vector;

    /// Constructs a non-nullable `FixedSizeList` storage dtype of 128 `element`s.
    fn fixed_size_list(element: DType) -> DType {
        DType::FixedSizeList(Arc::new(element), 128, Nullability::NonNullable)
    }

    /// A read fails unless the stored storage dtype is a `FixedSizeList` of non-nullable floats,
    /// and the error names the rejected dtype.
    #[rstest]
    #[case::non_fsl(
        DType::Primitive(PType::F32, Nullability::NonNullable),
        "Vector storage dtype must be a FixedSizeList, got f32"
    )]
    #[case::integer_elements(
        fixed_size_list(DType::Primitive(PType::U32, Nullability::NonNullable)),
        "Vector element type must be a float, got u32"
    )]
    #[case::non_primitive_elements(
        fixed_size_list(DType::Utf8(Nullability::NonNullable)),
        "Vector element dtype must be a float, got utf8"
    )]
    #[case::nullable_elements(
        fixed_size_list(DType::Primitive(PType::F32, Nullability::Nullable)),
        "vortex.tensor.vector storage must be fixed_size_list(f32)[128], got fixed_size_list(f32?)[128]"
    )]
    fn read_rejects_other_storage_dtype(#[case] storage: DType, #[case] expected: &str) {
        let result = Vector.deserialize(&[], storage);
        assert!(result.is_err_and(|error| error.to_string().contains(expected)));
    }

    #[test]
    fn list_contains_requires_eq() -> VortexResult<()> {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        vortex_functions::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let elements = Vector::try_new_vector_array(
            FixedSizeListArray::try_new(
                buffer![1.0f32, 2.0, 1.0, 3.0, 4.0, 5.0].into_array(),
                2,
                Validity::NonNullable,
                3,
            )?
            .into_array(),
        )?;
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let needle = Vector::try_new_vector_array(
            FixedSizeListArray::try_new(
                buffer![1.0f32, 3.0, 4.0, 6.0].into_array(),
                2,
                Validity::NonNullable,
                2,
            )?
            .into_array(),
        )?;

        assert_execution_paths(
            &[list, needle],
            |inputs| {
                ListContains::try_new(inputs[0].clone(), inputs[1].clone())
                    .map(IntoArray::into_array)
            },
            ExpectedOutcome::Array(BoolArray::from_iter([true, false]).into_array()),
            &mut ctx,
        );

        Ok(())
    }
}
