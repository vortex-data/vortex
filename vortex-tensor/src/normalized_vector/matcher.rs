// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::normalized_vector::NormalizedVector;
use crate::vector::VectorMatcherMetadata;

/// Matcher that accepts only the [`NormalizedVector`] extension type.
///
/// Use this when a consumer must reject plain [`Vector`](crate::vector::Vector) inputs. Callers
/// that can accept either should use [`AnyVector`](crate::vector::AnyVector) instead.
pub struct AnyNormalizedVector;

impl Matcher for AnyNormalizedVector {
    type Match<'a> = VectorMatcherMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if !ext_dtype.is::<NormalizedVector>() {
            return None;
        }

        let DType::FixedSizeList(element_dtype, list_size, _) = ext_dtype.storage_dtype() else {
            vortex_panic!(
                "`NormalizedVector` type somehow did not have a `FixedSizeList` storage type"
            )
        };
        assert!(element_dtype.is_float(), "element dtype must be float");
        assert!(
            !element_dtype.is_nullable(),
            "element dtype must be non-nullable"
        );

        let metadata = VectorMatcherMetadata::try_new(element_dtype.as_ptype(), *list_size, true)
            .vortex_expect("`NormalizedVector` type somehow did not have float elements");

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
    use crate::vector::AnyVector;
    use crate::vector::Vector;

    fn storage_dtype(element_ptype: PType, dimensions: u32) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(element_ptype, Nullability::NonNullable)),
            dimensions,
            Nullability::NonNullable,
        )
    }

    #[test]
    fn matches_normalized_vector_dtype() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage_dtype(PType::F32, 128))?
                .erased();

        let metadata = ext_dtype.metadata::<AnyNormalizedVector>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.dimensions(), 128);
        assert!(metadata.is_normalized());
        Ok(())
    }

    #[test]
    fn rejects_plain_vector() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype(PType::F32, 128))?.erased();

        assert!(ext_dtype.metadata_opt::<AnyNormalizedVector>().is_none());
        Ok(())
    }

    #[test]
    fn any_vector_matches_normalized_vector() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<NormalizedVector>::try_new(EmptyMetadata, storage_dtype(PType::F32, 128))?
                .erased();

        let metadata = ext_dtype.metadata::<AnyVector>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.dimensions(), 128);
        assert!(metadata.is_normalized());
        Ok(())
    }
}
