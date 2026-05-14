// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::types::vector::Vector;

pub struct AnyVector;

/// Convenience metadata for vectors.
///
/// Unlike `FixedShapeTensor`, the [`Vector`] type has `EmptyMetadata` as its metadata because all
/// of the important information is already stored in the dtype.
///
/// However, it is quite inconvenient to repeatedly unwrap the dtype to get the element type of the
/// vector and the number of dimensions.
///
/// Thus, we allow the matcher to return this metadata so that we can access this information more
/// easily.
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
        if !ext_dtype.is::<Vector>() {
            return None;
        }

        let DType::FixedSizeList(element_dtype, list_size, _) = ext_dtype.storage_dtype() else {
            vortex_panic!("`Vector` type somehow did not have a `FixedSizeList` storage type")
        };

        let dimensions = *list_size;

        assert!(element_dtype.is_float(), "element dtype must be float");
        assert!(
            !element_dtype.is_nullable(),
            "element dtype must be non-nullable"
        );
        let element_ptype = element_dtype.as_ptype();

        let vector_metadata = VectorMatcherMetadata::try_new(element_ptype, dimensions)
            .vortex_expect("`Vector` type somehow did not have float elements");

        Some(vector_metadata)
    }
}

impl VectorMatcherMetadata {
    /// Tries to create a new `VectorMatcherMetadata`.
    ///
    /// # Errors
    ///
    /// Returns an error if the element type is not a float.
    pub fn try_new(element_ptype: PType, dimensions: u32) -> VortexResult<Self> {
        vortex_ensure!(element_ptype.is_float());

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

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
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
