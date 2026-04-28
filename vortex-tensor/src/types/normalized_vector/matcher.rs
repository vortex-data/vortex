// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::types::normalized_vector::NormalizedVector;
use crate::types::vector::Vector;
use crate::types::vector::VectorMatcherMetadata;

/// Matcher that accepts only the [`NormalizedVector`] extension type.
///
/// Use this when a consumer requires the unit-norm guarantee. Callers that accept any
/// vector-shaped extension should use [`AnyTensor`](crate::matcher::AnyTensor).
pub struct AnyNormalizedVector;

impl Matcher for AnyNormalizedVector {
    type Match<'a> = VectorMatcherMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if !ext_dtype.is::<NormalizedVector>() {
            return None;
        }

        // `NormalizedVector` stores a `Vector(FixedSizeList<float, dim>)`. Drill into the inner
        // `Vector` to recover the dimension and element dtype.
        let DType::Extension(inner_ext) = ext_dtype.storage_dtype() else {
            vortex_panic!(
                "`NormalizedVector` storage must be `DType::Extension(Vector)`, \
                 got {}",
                ext_dtype.storage_dtype(),
            )
        };
        if !inner_ext.is::<Vector>() {
            vortex_panic!(
                "`NormalizedVector` inner extension must be `Vector`, got {}",
                inner_ext.id(),
            )
        }
        let DType::FixedSizeList(element_dtype, list_size, _) = inner_ext.storage_dtype() else {
            vortex_panic!(
                "inner `Vector` storage must be `FixedSizeList`, got {}",
                inner_ext.storage_dtype(),
            )
        };
        assert!(element_dtype.is_float(), "element dtype must be float");
        assert!(
            !element_dtype.is_nullable(),
            "element dtype must be non-nullable"
        );

        let metadata = VectorMatcherMetadata::try_new(element_dtype.as_ptype(), *list_size, true)
            .vortex_expect("`NormalizedVector` inner Vector did not have float elements");

        Some(metadata)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
    use vortex_error::VortexResult;

    use super::*;
    use crate::types::vector::AnyVector;
    use crate::types::vector::Vector;

    fn fsl_storage(element_ptype: PType, dimensions: u32) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(element_ptype, Nullability::NonNullable)),
            dimensions,
            Nullability::NonNullable,
        )
    }

    fn nv_storage(element_ptype: PType, dimensions: u32) -> VortexResult<DType> {
        let vector =
            ExtDType::<Vector>::try_new(EmptyMetadata, fsl_storage(element_ptype, dimensions))?
                .erased();
        Ok(DType::Extension(vector))
    }

    #[test]
    fn matches_normalized_vector_dtype() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<NormalizedVector>::try_new(EmptyMetadata, nv_storage(PType::F32, 128)?)?
                .erased();

        let metadata = ext_dtype.metadata::<AnyNormalizedVector>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.dimensions(), 128);
        Ok(())
    }

    #[test]
    fn rejects_plain_vector() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<Vector>::try_new(EmptyMetadata, fsl_storage(PType::F32, 128))?.erased();

        assert!(ext_dtype.metadata_opt::<AnyNormalizedVector>().is_none());
        Ok(())
    }

    #[test]
    fn any_vector_matches_normalized_vector() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<NormalizedVector>::try_new(EmptyMetadata, nv_storage(PType::F32, 128)?)?
                .erased();

        // `AnyVector` is the inclusive matcher: it matches both `Vector` and `NormalizedVector`.
        // Callers that need to distinguish the two should pair it with an
        // [`AnyNormalizedVector`] check, or use [`AnyTensor`](crate::matcher::AnyTensor) to also
        // accept `FixedShapeTensor`.
        let metadata = ext_dtype.metadata::<AnyVector>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.dimensions(), 128);
        Ok(())
    }
}
