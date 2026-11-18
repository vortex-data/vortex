// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Internal homes for tensor extension types.

pub mod fixed_shape_tensor;
pub mod vector;

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::compute::conformance::extension::test_ext_vtable_conformance;
    use vortex_array::dtype::Claim;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;

    use super::fixed_shape_tensor::FixedShapeTensor;
    use super::fixed_shape_tensor::FixedShapeTensorMetadata;
    use super::vector::Vector;
    use super::vector::VectorMetadata;
    use crate::tests::SESSION;

    fn assert_tensor_claim_conformance<V>(ext_dtype: ExtDType<V>) -> VortexResult<()>
    where
        V: for<'a> ExtVTable<NativeValue<'a> = &'a ScalarValue>,
    {
        let nan = f32::NAN;
        let other_nan = f32::from_bits(nan.to_bits() + 1);
        let elements = buffer![
            0.0f32, nan, 1.0, 2.0, // First row.
            0.0f32, nan, 1.0, 2.0, // Identical bits, including the NaN payload.
            -0.0f32, nan, 1.0, 2.0, // Different sign bit on zero.
            0.0f32, other_nan, 1.0, 2.0, // Different NaN payload.
        ];
        let storage =
            FixedSizeListArray::try_new(elements.into_array(), 4, Validity::NonNullable, 4)?
                .into_array();
        test_ext_vtable_conformance(
            &ext_dtype,
            &storage,
            Some(&|lhs, rhs| f32_element_bits(lhs).cmp(f32_element_bits(rhs))),
            &mut SESSION.create_execution_ctx(),
        );

        let dtype = DType::Extension(ext_dtype.erased());
        assert!(dtype.holds(Claim::Eq), "{dtype} must hold Eq");
        assert!(!dtype.holds(Claim::Ord), "{dtype} must lack Ord");

        Ok(())
    }

    // Only equality of the bit patterns matters for the Eq claim.
    fn f32_element_bits(value: &ScalarValue) -> impl Iterator<Item = u32> + '_ {
        value.as_list().iter().map(|element| match element {
            Some(ScalarValue::Primitive(PValue::F32(value))) => value.to_bits(),
            _ => vortex_panic!("the fixture uses non-null f32 elements"),
        })
    }

    #[test]
    fn vector_claims_eq_only() -> VortexResult<()> {
        let dtype =
            ExtDType::<Vector>::try_new(VectorMetadata::try_new(PType::F32, 4)?, NonNullable)?;
        assert_tensor_claim_conformance(dtype)
    }

    #[test]
    fn fixed_shape_tensor_claims_eq_only() -> VortexResult<()> {
        let metadata = FixedShapeTensorMetadata::new(PType::F32, vec![2, 2]);
        let dtype = ExtDType::<FixedShapeTensor>::try_new(metadata, NonNullable)?;
        assert_tensor_claim_conformance(dtype)
    }
}
