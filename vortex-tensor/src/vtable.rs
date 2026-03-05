// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::dtype::DType;
use vortex::dtype::extension::ExtId;
use vortex::dtype::extension::ExtVTable;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::scalar::ScalarValue;

use crate::FixedShapeTensor;
use crate::FixedShapeTensorMetadata;
use crate::proto;

impl ExtVTable for FixedShapeTensor {
    type Metadata = FixedShapeTensorMetadata;

    // TODO(connor): This is just a placeholder for now!!!
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new_ref("vortex.fixed_shape_tensor")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(proto::serialize(metadata))
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        proto::deserialize(metadata)
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        let DType::FixedSizeList(element_dtype, list_size, _nullability) = storage_dtype else {
            vortex_bail!(
                "FixedShapeTensor storage dtype must be a FixedSizeList, got {storage_dtype}"
            );
        };

        // Note that these constraints may be relaxed in the future.
        vortex_ensure!(
            element_dtype.is_primitive(),
            "FixedShapeTensor element dtype must be primitive, got {element_dtype} \
             (may change in the future)"
        );
        vortex_ensure!(
            !element_dtype.is_nullable(),
            "FixedShapeTensor element dtype must be non-nullable (may change in the future)"
        );

        let element_count: usize = metadata.logical_shape().iter().product();
        vortex_ensure_eq!(
            element_count,
            *list_size as usize,
            "FixedShapeTensor logical shape product ({element_count}) does not match \
             FixedSizeList size ({list_size})"
        );

        Ok(())
    }

    fn unpack_native<'a>(
        &self,
        _metadata: &'a Self::Metadata,
        _storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        // TODO(connor): This is just a placeholder. However, even if we have a dedicated native
        // type for a singular tensor, we do not need to validate anything as any backing memory
        // should be valid for a given tensor.
        Ok(storage_value)
    }
}

#[cfg(test)]
mod tests {
    use vortex::dtype::extension::ExtVTable;
    use vortex::error::VortexResult;

    use crate::FixedShapeTensor;
    use crate::FixedShapeTensorMetadata;

    fn assert_roundtrip(metadata: &FixedShapeTensorMetadata) -> VortexResult<()> {
        let vtable = FixedShapeTensor;
        let bytes = vtable.serialize_metadata(metadata)?;
        let deserialized = vtable.deserialize_metadata(&bytes)?;
        assert_eq!(&deserialized, metadata);
        Ok(())
    }

    #[test]
    fn roundtrip_shape_only() -> VortexResult<()> {
        assert_roundtrip(&FixedShapeTensorMetadata::new(vec![2, 3, 4]))
    }

    #[test]
    fn roundtrip_with_permutation() -> VortexResult<()> {
        assert_roundtrip(
            &FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![2, 0, 1])?,
        )
    }

    #[test]
    fn roundtrip_with_dim_names() -> VortexResult<()> {
        assert_roundtrip(
            &FixedShapeTensorMetadata::new(vec![3, 4])
                .with_dim_names(vec!["rows".into(), "cols".into()])?,
        )
    }

    #[test]
    fn roundtrip_all_fields() -> VortexResult<()> {
        assert_roundtrip(
            &FixedShapeTensorMetadata::new(vec![2, 3, 4])
                .with_dim_names(vec!["x".into(), "y".into(), "z".into()])?
                .with_permutation(vec![1, 2, 0])?,
        )
    }

    #[test]
    fn roundtrip_scalar_0d() -> VortexResult<()> {
        assert_roundtrip(&FixedShapeTensorMetadata::new(vec![]))
    }
}
