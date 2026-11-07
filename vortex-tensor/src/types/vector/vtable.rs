// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::types::vector::Vector;
use crate::types::vector::VectorMatcherMetadata;
use crate::types::vector::vector_metadata;

impl ExtVTable for Vector {
    type Metadata = VectorMatcherMetadata;

    // TODO(connor): This is just a placeholder for now.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.tensor.vector");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(
        &self,
        _metadata: &[u8],
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        vector_metadata(storage_dtype)
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_error::VortexResult;

    use crate::types::vector::Vector;
    use crate::types::vector::VectorMatcherMetadata;

    /// Constructs a `FixedSizeList` storage dtype with the given float [`PType`], list size, and
    /// [`Nullability`].
    fn vector_storage_dtype(ptype: PType, size: u32, nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(ptype, Nullability::NonNullable)),
            size,
            nullability,
        )
    }

    /// Constructs a non-nullable `FixedSizeList` storage dtype of 128 `element`s.
    fn fixed_size_list(element: DType) -> DType {
        DType::FixedSizeList(Arc::new(element), 128, Nullability::NonNullable)
    }

    #[rstest]
    #[case::f16(PType::F16)]
    #[case::f32(PType::F32)]
    #[case::f64(PType::F64)]
    fn read_accepts_float_types(#[case] ptype: PType) -> VortexResult<()> {
        let storage = vector_storage_dtype(ptype, 128, Nullability::NonNullable);
        Vector.deserialize(&[], storage)?;
        Ok(())
    }

    #[rstest]
    #[case::nullable(Nullability::Nullable)]
    #[case::non_nullable(Nullability::NonNullable)]
    fn read_accepts_any_outer_nullability(#[case] nullability: Nullability) -> VortexResult<()> {
        let storage = vector_storage_dtype(PType::F32, 128, nullability);
        Vector.deserialize(&[], storage)?;
        Ok(())
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
    fn roundtrip_metadata() -> VortexResult<()> {
        let vtable = Vector;
        let metadata = VectorMatcherMetadata::try_new(PType::F32, 128)?;
        let bytes = vtable.serialize_metadata(&metadata)?;
        assert!(bytes.is_empty());
        let storage = vector_storage_dtype(PType::F32, 128, Nullability::NonNullable);
        let deserialized = vtable.deserialize_metadata(&bytes, &storage)?;
        assert_eq!(deserialized, metadata);
        Ok(())
    }
}
