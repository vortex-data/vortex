// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::types::fixed_shape_tensor::FixedShapeTensor;
use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;
use crate::types::fixed_shape_tensor::proto;

impl ExtVTable for FixedShapeTensor {
    type Metadata = FixedShapeTensorMetadata;

    // TODO(connor): This is just a placeholder for now!!!
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.tensor.fixed_shape_tensor");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(proto::serialize(metadata))
    }

    fn deserialize_metadata(
        &self,
        metadata: &[u8],
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        let DType::FixedSizeList(element_dtype, ..) = storage_dtype else {
            vortex_bail!(
                "FixedShapeTensor storage dtype must be a FixedSizeList, got {storage_dtype}"
            );
        };
        let DType::Primitive(element_ptype, _) = element_dtype.as_ref() else {
            vortex_bail!("FixedShapeTensor element dtype must be primitive, got {element_dtype}");
        };

        proto::deserialize(metadata, *element_ptype)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        let list_size = flat_list_size(metadata).vortex_expect(
            "FixedShapeTensor::validate_metadata checks that the list size fits u32",
        );
        DType::FixedSizeList(
            Arc::new(DType::Primitive(
                metadata.element_ptype(),
                Nullability::NonNullable,
            )),
            list_size,
            Nullability::NonNullable,
        )
    }

    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        vortex_ensure!(
            flat_list_size(metadata).is_some(),
            "FixedShapeTensor dimensions and their product must fit u32, got shape {:?}",
            metadata.logical_shape()
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

/// The `FixedSizeList` size of the storage, which is the product of the logical shape. It is
/// `None` when a dimension or the product does not fit `u32`.
///
/// Each dimension must fit `u32` on its own, because the serialized metadata stores dimensions as
/// `u32`. A zero dimension makes the product zero, even when the product of the other dimensions
/// does not fit `u32`.
fn flat_list_size(metadata: &FixedShapeTensorMetadata) -> Option<u32> {
    let shape = metadata
        .logical_shape()
        .iter()
        .map(|&dim| u32::try_from(dim).ok())
        .collect::<Option<Vec<u32>>>()?;

    if shape.contains(&0) {
        return Some(0);
    }

    shape.into_iter().try_fold(1, u32::checked_mul)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_error::VortexResult;

    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

    /// Serializes the given metadata and deserializes it with its declared storage dtype, asserting
    /// equality.
    fn assert_roundtrip(metadata: &FixedShapeTensorMetadata) -> VortexResult<()> {
        let vtable = FixedShapeTensor;
        let bytes = vtable.serialize_metadata(metadata)?;
        let deserialized =
            vtable.deserialize_metadata(&bytes, &FixedShapeTensor::storage_dtype(metadata))?;
        assert_eq!(&deserialized, metadata);
        Ok(())
    }

    /// Constructs a non-nullable `FixedSizeList` storage dtype of `size` `element`s.
    fn fixed_size_list(element: DType, size: u32) -> DType {
        DType::FixedSizeList(Arc::new(element), size, Nullability::NonNullable)
    }

    #[rstest]
    #[case::scalar_0d(FixedShapeTensorMetadata::new(PType::F32, vec![]))]
    #[case::shape_only(FixedShapeTensorMetadata::new(PType::F64, vec![2, 3, 4]))]
    fn roundtrip_simple(#[case] metadata: FixedShapeTensorMetadata) -> VortexResult<()> {
        assert_roundtrip(&metadata)
    }

    #[rstest]
    #[case::with_permutation(
        FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4])
            .with_permutation(vec![2, 0, 1])
    )]
    #[case::with_dim_names(
        FixedShapeTensorMetadata::new(PType::F32, vec![3, 4])
            .with_dim_names(vec!["rows".into(), "cols".into()])
    )]
    #[case::all_fields(
        FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()])
            .and_then(|m| m.with_permutation(vec![1, 2, 0]))
    )]
    fn roundtrip_with_options(
        #[case] metadata: VortexResult<FixedShapeTensorMetadata>,
    ) -> VortexResult<()> {
        assert_roundtrip(&metadata?)
    }

    /// The serialized metadata is the protobuf encoding of the shape, the dimension names, and the
    /// permutation. Files store these bytes, so they must not change.
    #[test]
    fn serialized_metadata_bytes() -> VortexResult<()> {
        let metadata = FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()])?
            .with_permutation(vec![1, 2, 0])?;

        let expected = [
            0x0a, 3, 2, 3, 4, // Field 1 holds the packed logical shape.
            0x12, 1, b'x', // Field 2 holds the first dimension name.
            0x12, 1, b'y', // Field 2 holds the second dimension name.
            0x12, 1, b'z', // Field 2 holds the third dimension name.
            0x1a, 3, 1, 2, 0, // Field 3 holds the packed permutation.
        ];
        assert_eq!(FixedShapeTensor.serialize_metadata(&metadata)?, expected);

        Ok(())
    }

    /// A read fails unless the stored storage dtype is the declared one, and the error names the
    /// rejected dtype. The element type comes from the stored storage dtype.
    #[rstest]
    #[case::non_fsl(
        DType::Primitive(PType::F32, Nullability::NonNullable),
        "FixedShapeTensor storage dtype must be a FixedSizeList, got f32"
    )]
    #[case::non_primitive_elements(
        fixed_size_list(DType::Utf8(Nullability::NonNullable), 6),
        "FixedShapeTensor element dtype must be primitive, got utf8"
    )]
    #[case::wrong_list_size(
        fixed_size_list(DType::Primitive(PType::F32, Nullability::NonNullable), 5),
        "vortex.tensor.fixed_shape_tensor storage must be fixed_size_list(f32)[6], got fixed_size_list(f32)[5]"
    )]
    #[case::nullable_elements(
        fixed_size_list(DType::Primitive(PType::F32, Nullability::Nullable), 6),
        "vortex.tensor.fixed_shape_tensor storage must be fixed_size_list(f32)[6], got fixed_size_list(f32?)[6]"
    )]
    fn read_rejects_other_storage_dtype(
        #[case] storage_dtype: DType,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        let metadata = FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]);
        let bytes = FixedShapeTensor.serialize_metadata(&metadata)?;

        let result = FixedShapeTensor.deserialize(&bytes, storage_dtype);
        assert!(result.is_err_and(|error| error.to_string().contains(expected)));

        Ok(())
    }

    /// A shape is rejected when a dimension or the product of the dimensions does not fit the
    /// `u32` list size of the storage.
    #[rstest]
    #[case::product_over_u32(vec![1 << 16, 1 << 16])]
    #[case::dimension_over_u32(vec![0, 1 << 32])]
    fn rejects_shape_over_u32(#[case] shape: Vec<usize>) {
        let metadata = FixedShapeTensorMetadata::new(PType::F32, shape);
        assert!(ExtDType::<FixedShapeTensor>::try_new(metadata, Nullability::NonNullable).is_err());
    }

    /// A zero dimension makes the list size zero, even when the product of the dimensions before
    /// it does not fit `u32`, so a tensor of that shape still reads.
    #[test]
    fn reads_zero_dimension_after_overflowing_dimensions() -> VortexResult<()> {
        let metadata = FixedShapeTensorMetadata::new(PType::F32, vec![1 << 16, 1 << 16, 0]);
        let bytes = FixedShapeTensor.serialize_metadata(&metadata)?;

        let storage_dtype =
            fixed_size_list(DType::Primitive(PType::F32, Nullability::NonNullable), 0);
        FixedShapeTensor.deserialize(&bytes, storage_dtype)?;

        Ok(())
    }
}
