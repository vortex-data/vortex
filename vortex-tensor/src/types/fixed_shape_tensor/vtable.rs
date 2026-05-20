// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::types::fixed_shape_tensor::FixedShapeTensor;
use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;
use crate::types::fixed_shape_tensor::proto;

impl ExtVTable for FixedShapeTensor {
    type Metadata = FixedShapeTensorMetadata;

    // TODO(connor): This is just a placeholder for now!!!
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.tensor.fixed_shape_tensor")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(proto::serialize(metadata))
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        proto::deserialize(metadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        let storage_dtype = ext_dtype.storage_dtype();
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

        let element_count: usize = ext_dtype.metadata().logical_shape().iter().product();
        vortex_ensure_eq!(
            element_count,
            *list_size as usize,
            "FixedShapeTensor logical shape product ({element_count}) does not match \
             FixedSizeList size ({list_size})"
        );

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
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
    use rstest::rstest;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_error::VortexResult;

    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

    /// Serializes and deserializes the given metadata through protobuf, asserting equality.
    fn assert_roundtrip(metadata: &FixedShapeTensorMetadata) -> VortexResult<()> {
        let vtable = FixedShapeTensor;
        let bytes = vtable.serialize_metadata(metadata)?;
        let deserialized = vtable.deserialize_metadata(&bytes)?;
        assert_eq!(&deserialized, metadata);
        Ok(())
    }

    #[rstest]
    #[case::scalar_0d(FixedShapeTensorMetadata::new(vec![]))]
    #[case::shape_only(FixedShapeTensorMetadata::new(vec![2, 3, 4]))]
    fn roundtrip_simple(#[case] metadata: FixedShapeTensorMetadata) -> VortexResult<()> {
        assert_roundtrip(&metadata)
    }

    #[rstest]
    #[case::with_permutation(
        FixedShapeTensorMetadata::new(vec![2, 3, 4])
            .with_permutation(vec![2, 0, 1])
    )]
    #[case::with_dim_names(
        FixedShapeTensorMetadata::new(vec![3, 4])
            .with_dim_names(vec!["rows".into(), "cols".into()])
    )]
    #[case::all_fields(
        FixedShapeTensorMetadata::new(vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()])
            .and_then(|m| m.with_permutation(vec![1, 2, 0]))
    )]
    fn roundtrip_with_options(
        #[case] metadata: VortexResult<FixedShapeTensorMetadata>,
    ) -> VortexResult<()> {
        assert_roundtrip(&metadata?)
    }
}
