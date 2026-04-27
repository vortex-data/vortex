// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::types::vector::Vector;
use crate::types::vector::validate_vector_storage_dtype;

/// Vortex extension id for [`Vector`].
pub(crate) static ID: CachedId = CachedId::new("vortex.tensor.vector");

impl ExtVTable for Vector {
    type Metadata = EmptyMetadata;

    // TODO(connor): This is just a placeholder for now.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn least_supertype(ext_dtype: &ExtDType<Self>, other: &DType) -> Option<DType> {
        let DType::Extension(other_ext) = other else {
            return None;
        };
        if !other_ext.is::<Self>() {
            return None;
        }
        let widened = ext_dtype
            .storage_dtype()
            .least_supertype(other_ext.storage_dtype())?;
        let ext = ExtDType::<Self>::try_new(EmptyMetadata, widened).ok()?;
        Some(DType::Extension(ext.erased()))
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

    use crate::types::vector::Vector;

    /// Constructs a `FixedSizeList` storage dtype with the given float [`PType`], list size, and
    /// [`Nullability`].
    fn vector_storage_dtype(ptype: PType, size: u32, nullability: Nullability) -> DType {
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
        let storage = vector_storage_dtype(ptype, 128, Nullability::NonNullable);
        ExtDType::<Vector>::try_new(EmptyMetadata, storage)?;
        Ok(())
    }

    #[rstest]
    #[case::nullable(Nullability::Nullable)]
    #[case::non_nullable(Nullability::NonNullable)]
    fn validate_accepts_any_outer_nullability(
        #[case] nullability: Nullability,
    ) -> VortexResult<()> {
        let storage = vector_storage_dtype(PType::F32, 128, nullability);
        ExtDType::<Vector>::try_new(EmptyMetadata, storage)?;
        Ok(())
    }

    #[test]
    fn validate_rejects_non_fsl() {
        let storage = DType::Primitive(PType::F32, Nullability::NonNullable);
        assert!(ExtDType::<Vector>::try_new(EmptyMetadata, storage).is_err());
    }

    #[test]
    fn validate_rejects_integer_elements() {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
            128,
            Nullability::NonNullable,
        );
        assert!(ExtDType::<Vector>::try_new(EmptyMetadata, storage).is_err());
    }

    #[test]
    fn validate_rejects_nullable_elements() {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::Nullable)),
            128,
            Nullability::NonNullable,
        );
        assert!(ExtDType::<Vector>::try_new(EmptyMetadata, storage).is_err());
    }

    #[test]
    fn roundtrip_metadata() -> VortexResult<()> {
        let vtable = Vector;
        let bytes = vtable.serialize_metadata(&EmptyMetadata)?;
        assert!(bytes.is_empty());
        let deserialized = vtable.deserialize_metadata(&bytes)?;
        assert_eq!(deserialized, EmptyMetadata);
        Ok(())
    }

    fn vector_dtype(ptype: PType, dims: u32, outer: Nullability) -> DType {
        let storage = vector_storage_dtype(ptype, dims, outer);
        DType::Extension(
            ExtDType::<Vector>::try_new(EmptyMetadata, storage)
                .unwrap()
                .erased(),
        )
    }

    #[rstest]
    #[case::widens_float_precision(
        vector_dtype(PType::F32, 768, Nullability::NonNullable),
        vector_dtype(PType::F64, 768, Nullability::NonNullable),
        Some(vector_dtype(PType::F64, 768, Nullability::NonNullable))
    )]
    #[case::dim_mismatch_returns_none(
        vector_dtype(PType::F32, 768, Nullability::NonNullable),
        vector_dtype(PType::F32, 1024, Nullability::NonNullable),
        None
    )]
    #[case::vs_non_extension_returns_none(
        vector_dtype(PType::F32, 768, Nullability::NonNullable),
        DType::Primitive(PType::F32, Nullability::NonNullable),
        None
    )]
    #[case::unions_outer_nullability_with_float_widening(
        vector_dtype(PType::F32, 4, Nullability::NonNullable),
        vector_dtype(PType::F64, 4, Nullability::Nullable),
        Some(vector_dtype(PType::F64, 4, Nullability::Nullable))
    )]
    #[case::same_ptype_unions_outer_nullability(
        vector_dtype(PType::F32, 4, Nullability::NonNullable),
        vector_dtype(PType::F32, 4, Nullability::Nullable),
        Some(vector_dtype(PType::F32, 4, Nullability::Nullable))
    )]
    fn vector_least_supertype(
        #[case] lhs: DType,
        #[case] rhs: DType,
        #[case] expected: Option<DType>,
    ) {
        assert_eq!(lhs.least_supertype(&rhs), expected);
    }
}
