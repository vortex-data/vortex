// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::fixed_shape::FixedShapeTensor;
use crate::fixed_shape::FixedShapeTensorMetadata;

pub struct AnyFixedShapeTensor;

/// Convenience metadata for fixed-shape tensors.
///
/// Fixed-shape tensors already store their logical metadata directly, but callers also often need
/// the flattened storage list size and element primitive type from the storage dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedShapeTensorMatcherMetadata<'a> {
    /// The logical fixed-shape tensor metadata stored on the extension dtype.
    metadata: &'a FixedShapeTensorMetadata,

    /// The primitive element type of the tensor storage.
    ///
    /// Fixed-shape tensors currently require non-nullable primitive elements.
    element_ptype: PType,

    /// The flattened element count for each tensor row in storage order.
    ///
    /// This matches the `FixedSizeList` list size in the storage dtype, which is the product of
    /// the logical shape dimensions.
    flat_list_size: u32,
}

impl Matcher for AnyFixedShapeTensor {
    type Match<'a> = FixedShapeTensorMatcherMetadata<'a>;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if !ext_dtype.is::<FixedShapeTensor>() {
            return None;
        }

        let metadata = ext_dtype
            .metadata_opt::<FixedShapeTensor>()
            .vortex_expect("`FixedShapeTensor` type somehow did not have metadata");

        let DType::FixedSizeList(element_dtype, list_size, _) = ext_dtype.storage_dtype() else {
            vortex_panic!(
                "`FixedShapeTensor` type somehow did not have a `FixedSizeList` storage type"
            )
        };

        assert!(
            element_dtype.is_primitive(),
            "element dtype must be primitive"
        );
        assert!(
            !element_dtype.is_nullable(),
            "element dtype must be non-nullable"
        );

        Some(FixedShapeTensorMatcherMetadata {
            metadata,
            element_ptype: element_dtype.as_ptype(),
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
        self.element_ptype
    }

    /// Returns the flattened element count for each tensor row.
    pub fn flat_list_size(&self) -> u32 {
        self.flat_list_size
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
    use crate::vector::Vector;

    fn tensor_storage_dtype(element_ptype: PType, list_size: u32) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(element_ptype, Nullability::NonNullable)),
            list_size,
            Nullability::NonNullable,
        )
    }

    #[test]
    fn matches_fixed_shape_tensor_dtype_metadata() -> VortexResult<()> {
        let ext_dtype = ExtDType::<FixedShapeTensor>::try_new(
            FixedShapeTensorMetadata::new(vec![2, 3, 4]),
            tensor_storage_dtype(PType::F32, 24),
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
        let ext_dtype =
            ExtDType::<Vector>::try_new(EmptyMetadata, tensor_storage_dtype(PType::F32, 24))?
                .erased();

        assert!(ext_dtype.metadata_opt::<AnyFixedShapeTensor>().is_none());
        Ok(())
    }
}
