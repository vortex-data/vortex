// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;

use crate::normalized_vector::NormalizedVector;
use crate::vector::validate_vector_storage_dtype;

impl ExtVTable for NormalizedVector {
    type Metadata = EmptyMetadata;

    // TODO(connor): This is just a placeholder for now.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.tensor.normalized_vector")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        validate_vector_storage_dtype(ext_dtype.storage_dtype())
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
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::extension::EmptyMetadata;
    use vortex_error::VortexResult;

    use crate::normalized_vector::NormalizedVector;

    fn storage_dtype(ptype: PType, size: u32, nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(ptype, Nullability::NonNullable)),
            size,
            nullability,
        )
    }

    #[rstest]
    #[case::f16(PType::F16)]
    #[case::f32(PType::F32)]
    #[case::f64(PType::F64)]
    fn validate_accepts_float_types(#[case] ptype: PType) -> VortexResult<()> {
        let storage = storage_dtype(ptype, 64, Nullability::NonNullable);
        ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage)?;
        Ok(())
    }

    #[rstest]
    #[case::nullable(Nullability::Nullable)]
    #[case::non_nullable(Nullability::NonNullable)]
    fn validate_accepts_any_outer_nullability(
        #[case] nullability: Nullability,
    ) -> VortexResult<()> {
        let storage = storage_dtype(PType::F32, 64, nullability);
        ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage)?;
        Ok(())
    }

    #[test]
    fn validate_rejects_non_fsl() {
        let storage = DType::Primitive(PType::F32, Nullability::NonNullable);
        assert!(ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage).is_err());
    }

    #[test]
    fn validate_rejects_integer_elements() {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
            64,
            Nullability::NonNullable,
        );
        assert!(ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage).is_err());
    }

    #[test]
    fn validate_rejects_nullable_elements() {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::Nullable)),
            64,
            Nullability::NonNullable,
        );
        assert!(ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage).is_err());
    }

    #[test]
    fn roundtrip_metadata() -> VortexResult<()> {
        let vtable = NormalizedVector;
        let bytes = vtable.serialize_metadata(&EmptyMetadata)?;
        assert!(bytes.is_empty());
        let deserialized = vtable.deserialize_metadata(&bytes)?;
        assert_eq!(deserialized, EmptyMetadata);
        Ok(())
    }
}
