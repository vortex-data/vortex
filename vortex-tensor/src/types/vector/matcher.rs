// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::types::unit_vector::UnitVector;
use crate::types::vector::Vector;

/// Matches both [`Vector`] and [`UnitVector`].
pub struct AnyVector;

/// Shape metadata derived from [`Vector`] or [`UnitVector`] fixed-size-list storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorMatcherMetadata {
    /// The floating-point element ptype.
    element_ptype: PType,

    /// The number of elements in each vector row.
    dimensions: u32,
}

impl Matcher for AnyVector {
    type Match<'a> = VectorMatcherMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if !ext_dtype.is::<Vector>() && !ext_dtype.is::<UnitVector>() {
            return None;
        }

        Some(match_vector_storage(ext_dtype))
    }
}

pub(crate) fn match_vector_storage(ext_dtype: &ExtDTypeRef) -> VectorMatcherMetadata {
    let DType::FixedSizeList(element_dtype, list_size, _) = ext_dtype.storage_dtype() else {
        vortex_panic!("vector dtype somehow did not have a `FixedSizeList` storage type")
    };

    VectorMatcherMetadata {
        element_ptype: element_dtype.as_ptype(),
        dimensions: *list_size,
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
            "vector element ptype must be floating point, got {element_ptype}",
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::EmptyMetadata;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::*;
    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

    fn vector_storage_dtype(element_ptype: PType, dimensions: u32) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(element_ptype, Nullability::NonNullable)),
            dimensions,
            Nullability::NonNullable,
        )
    }

    #[test]
    fn matches_vector_dtype_metadata() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::<Vector>::try_new(EmptyMetadata, vector_storage_dtype(PType::F32, 256))?
                .erased();

        let metadata = ext_dtype.metadata::<AnyVector>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.dimensions(), 256);
        Ok(())
    }

    #[test]
    fn does_not_match_fixed_shape_tensor() -> VortexResult<()> {
        let ext_dtype = ExtDType::<FixedShapeTensor>::try_new(
            FixedShapeTensorMetadata::new(vec![16, 16]),
            vector_storage_dtype(PType::F32, 256),
        )?
        .erased();

        assert!(ext_dtype.metadata_opt::<AnyVector>().is_none());
        Ok(())
    }
}
