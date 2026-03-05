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

impl ExtVTable for FixedShapeTensor {
    type Metadata = FixedShapeTensorMetadata;

    // TODO(connor): This is just a placeholder for now!!!
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new_ref("vortex.tensor")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        todo!()
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        todo!()
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
