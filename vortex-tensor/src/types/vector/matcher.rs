// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matching [`Vector`] extension dtypes.
//!
//! The matcher returns the type's metadata without inspecting its declared storage.

use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;

use crate::types::vector::Vector;
use crate::types::vector::VectorMetadata;

/// Matches a [`Vector`] dtype and returns its metadata.
pub struct AnyVector;

impl Matcher for AnyVector {
    type Match<'a> = VectorMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        ext_dtype.metadata_opt::<Vector>().copied()
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::*;
    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

    #[test]
    fn matches_vector_dtype_metadata() -> VortexResult<()> {
        let metadata = VectorMetadata::try_new(PType::F32, 256)?;
        let ext_dtype = ExtDType::<Vector>::try_new(metadata, Nullability::NonNullable)?.erased();

        let metadata = ext_dtype.metadata::<AnyVector>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.dimensions(), 256);
        Ok(())
    }

    #[test]
    fn does_not_match_fixed_shape_tensor() -> VortexResult<()> {
        let ext_dtype = ExtDType::<FixedShapeTensor>::try_new(
            FixedShapeTensorMetadata::new(PType::F32, vec![16, 16]),
            Nullability::NonNullable,
        )?
        .erased();

        assert!(ext_dtype.metadata_opt::<AnyVector>().is_none());
        Ok(())
    }
}
