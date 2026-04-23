// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtRefinedSource;
use vortex_array::dtype::extension::RefinementVTable;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;

use crate::types::normalized_vector::NormalizedVector;
use crate::types::normalized_vector::validate_unit_norm_rows;
use crate::types::vector::Vector;

impl RefinementVTable for NormalizedVector {
    type Source = ExtRefinedSource<Vector>;
    type Metadata = EmptyMetadata;

    // TODO(connor): This is just a placeholder for now. The per-scalar refinement is a
    // no-op; unit-norm is enforced at array construction via [`validate_array`].
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.tensor.normalized_vector")
    }

    fn refine_scalar<'a>(
        _metadata: &'a Self::Metadata,
        source_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        // Per-scalar refinement is a no-op: unit-norm is enforced at array construction via
        // [`validate_array`], matching how L2Denorm validates up front rather than on each
        // scalar access.
        Ok(source_value)
    }

    fn validate_array(_metadata: &Self::Metadata, source_array: &ArrayRef) -> VortexResult<()> {
        // `source_array` is a `Vector` extension array (`ExtRefinedSource<Vector>::Value`).
        let mut ctx = vortex_array::LEGACY_SESSION.create_execution_ctx();
        validate_unit_norm_rows(source_array, &mut ctx)
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
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

    use crate::types::normalized_vector::NormalizedVector;
    use crate::types::vector::Vector;

    /// The NormalizedVector storage dtype is `DType::Extension(Vector(FSL<float, dim>))`.
    fn nv_storage_dtype(ptype: PType, size: u32, nullability: Nullability) -> VortexResult<DType> {
        let fsl = DType::FixedSizeList(
            Arc::new(DType::Primitive(ptype, Nullability::NonNullable)),
            size,
            nullability,
        );
        let vector = ExtDType::<Vector>::try_new(EmptyMetadata, fsl)?.erased();
        Ok(DType::Extension(vector))
    }

    #[rstest]
    #[case::f16(PType::F16)]
    #[case::f32(PType::F32)]
    #[case::f64(PType::F64)]
    fn validate_accepts_float_types(#[case] ptype: PType) -> VortexResult<()> {
        let storage = nv_storage_dtype(ptype, 64, Nullability::NonNullable)?;
        ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage)?;
        Ok(())
    }

    #[rstest]
    #[case::nullable(Nullability::Nullable)]
    #[case::non_nullable(Nullability::NonNullable)]
    fn validate_accepts_any_outer_nullability(
        #[case] nullability: Nullability,
    ) -> VortexResult<()> {
        let storage = nv_storage_dtype(PType::F32, 64, nullability)?;
        ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage)?;
        Ok(())
    }

    #[test]
    fn validate_rejects_non_extension_storage() {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
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
