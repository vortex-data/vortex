// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::types::vector::Vector;

pub struct AnyVector;

/// The metadata of the [`Vector`] type: its element type and number of dimensions.
///
/// The serialized metadata is empty, because the storage dtype holds both facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorMatcherMetadata {
    /// The element type of the vectors. Note that vector elements are _always_ non-nullable.
    ///
    /// This MUST be a floating point type (f16, f32, f64).
    element_ptype: PType,

    /// The number of dimensions of the vector. This is always fixed.
    dimensions: u32,
}

impl Matcher for AnyVector {
    type Match<'a> = VectorMatcherMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        ext_dtype.metadata_opt::<Vector>().copied()
    }
}

impl VectorMatcherMetadata {
    /// Tries to create a new `VectorMatcherMetadata`.
    ///
    /// # Errors
    ///
    /// Returns an error if the element type is not a float.
    pub fn try_new(element_ptype: PType, dimensions: u32) -> VortexResult<Self> {
        vortex_ensure!(
            element_ptype.is_float(),
            "Vector element type must be a float, got {element_ptype}"
        );

        Ok(Self {
            element_ptype,
            dimensions,
        })
    }

    /// Returns the element type of the vectors.
    pub fn element_ptype(&self) -> PType {
        self.element_ptype
    }

    /// Returns the number of dimensions of the vector.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }
}

/// Prints nothing, like the serialized metadata, because a printed dtype already shows both facts
/// in its storage dtype.
impl fmt::Display for VectorMatcherMetadata {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
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
        let metadata = VectorMatcherMetadata::try_new(PType::F32, 256)?;
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
