// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Internal homes for tensor extension types.

pub mod fixed_shape_tensor;
pub mod vector;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::Claim;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtDTypeRef;
    use vortex_error::VortexResult;

    use super::fixed_shape_tensor::FixedShapeTensor;
    use super::fixed_shape_tensor::FixedShapeTensorMetadata;
    use super::vector::Vector;
    use super::vector::VectorMetadata;

    #[rstest]
    #[case::vector(VectorMetadata::try_new(PType::F32, 3)
        .and_then(|metadata| ExtDType::<Vector>::try_new(metadata, NonNullable))
        .map(ExtDType::erased))]
    #[case::fixed_shape_tensor(ExtDType::<FixedShapeTensor>::try_new(
        FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]), NonNullable,
    ).map(ExtDType::erased))]
    fn vector_and_tensor_claim_eq_only(
        #[case] ext_dtype: VortexResult<ExtDTypeRef>,
    ) -> VortexResult<()> {
        let dtype = DType::Extension(ext_dtype?);

        assert!(dtype.holds(Claim::Eq), "{dtype} must hold Eq");
        assert!(!dtype.holds(Claim::Ord), "{dtype} must lack Ord");

        Ok(())
    }
}
