// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::vortex_panic;

use crate::types::fixed_shape_tensor::FixedShapeTensor;
use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

/// Matches fixed-shape tensor dtypes and borrows their metadata.
pub struct AnyFixedShapeTensor;

/// A matched view of [`FixedShapeTensorMetadata`] and its flattened storage list size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedShapeTensorMatcherMetadata<'a> {
    /// The logical fixed-shape tensor metadata stored on the extension dtype.
    metadata: &'a FixedShapeTensorMetadata,
    /// The flattened element count for each tensor row in storage order.
    ///
    /// This matches the `FixedSizeList` list size in the storage dtype, which is the product of
    /// the logical shape dimensions.
    flat_list_size: u32,
}

impl Matcher for AnyFixedShapeTensor {
    type Match<'a> = FixedShapeTensorMatcherMetadata<'a>;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        let metadata = ext_dtype.metadata_opt::<FixedShapeTensor>()?;

        let DType::FixedSizeList(_, list_size, _) = ext_dtype.storage_dtype() else {
            vortex_panic!("FixedShapeTensor::storage_dtype declares FixedSizeList storage")
        };

        Some(FixedShapeTensorMatcherMetadata {
            metadata,
            flat_list_size: *list_size,
        })
    }
}

impl FixedShapeTensorMatcherMetadata<'_> {
    /// Returns the underlying fixed-shape tensor metadata.
    pub fn metadata(&self) -> &FixedShapeTensorMetadata {
        self.metadata
    }

    /// Returns the tensor element type.
    pub fn element_ptype(&self) -> PType {
        self.metadata.element_ptype()
    }

    /// Returns the flattened element count for each tensor row.
    pub fn flat_list_size(&self) -> u32 {
        self.flat_list_size
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::*;
    use crate::types::vector::Vector;
    use crate::types::vector::VectorMetadata;

    #[test]
    fn matches_fixed_shape_tensor_dtype_metadata() -> VortexResult<()> {
        let ext_dtype = ExtDType::<FixedShapeTensor>::try_new(
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4]),
            Nullability::NonNullable,
        )?
        .erased();

        let metadata = ext_dtype.metadata::<AnyFixedShapeTensor>();
        assert_eq!(metadata.element_ptype(), PType::F32);
        assert_eq!(metadata.flat_list_size(), 24);
        assert_eq!(metadata.metadata().logical_shape(), &[2, 3, 4]);
        Ok(())
    }

    #[test]
    fn does_not_match_vector() -> VortexResult<()> {
        let metadata = VectorMetadata::try_new(PType::F32, 24)?;
        let ext_dtype = ExtDType::<Vector>::try_new(metadata, Nullability::NonNullable)?.erased();

        assert!(ext_dtype.metadata_opt::<AnyFixedShapeTensor>().is_none());
        Ok(())
    }
}
